"""Load test for the LOCAL Learning Plan Generator - one real browser per user.

Streamlit talks to the browser over a WebSocket and re-runs the whole script on
every widget change, so an HTTP load generator would measure nothing: there are
no endpoints to hit. Each virtual user therefore drives a real headless Chromium
through the journey a trainer actually takes.

    Load app -> pick a programme (fetches + indexes the OS and the Curriculum)
             -> pick a unit from the matched table
             -> fill the plan details
             -> Generate Learning Plan

Run the app with LOADTEST_MOCK=1 (see ai_client.loadtest_mock_enabled). Without
it every user queues behind the same Groq per-minute token allowance and the
run measures Groq's rate limiter instead of this app.

    one user, visible output:   python locustfile.py
    the real run, headless:     locust -f locustfile.py --host http://localhost:8501 \
                                  --headless -u 30 -r 0.1 -t 10m \
                                  --html results/report.html --csv results/stats
"""

from __future__ import annotations

import asyncio
import random

import gevent
from locust import between, events, task
from locust_plugins.users import playwright as pwmod
from locust_plugins.users.playwright import PlaywrightUser, event, pw
from playwright.async_api import async_playwright

# --------------------------------------------------------------------------- #
# Windows: Playwright's event loop needs a real thread
# --------------------------------------------------------------------------- #
# locust-plugins starts it with `gevent.spawn(loop.run_forever)` - a greenlet,
# not a thread. On Linux gevent's patched selectors make that cooperative; on
# Windows asyncio runs an IOCP loop that gevent cannot patch, so run_forever
# never yields, the hub stops scheduling every other greenlet, and each user
# hangs inside __init__ before its first task ever starts. The symptom is
# "Ramping to 1 users" followed by silence.
#
# So the loop goes on a genuine OS thread. It is started through the raw
# `_thread` primitive gevent saved before patching, NOT through threading.Thread:
# Thread.start() waits on an Event, and an unpatched Event waited on from a
# patched main thread deadlocks just as surely. Everything the plugin does
# reaches the loop through asyncio.run_coroutine_threadsafe, which is built to
# cross threads, so nothing else here has to change.
_start_new_thread = gevent.monkey.get_original("_thread", "start_new_thread")

events.test_start.remove_listener(pwmod.on_start)
# `os.system("reset")` on quit is a Unix habit that only prints an error here.
events.quitting.remove_listener(pwmod.on_locust_quit)


@events.test_start.add_listener
def _playwright_loop_on_a_real_thread(environment, **kwargs) -> None:
    pwmod.loop = asyncio.new_event_loop()
    _start_new_thread(pwmod.loop.run_forever, ())

# =========================================================================== #
# Settings - the real labels, read off the running app (every Streamlit widget
# puts its label on the input as aria-label, which is what these match).
# =========================================================================== #

# A programme whose folder holds both documents and whose PDFs are already in
# .drive_cache, so the run measures this app's parsing rather than Google's
# bandwidth. Any programme in the library works; this one has 17 matched units.
PROGRAMME = "ICT Technician Level 6"

SELECTBOXES = {"Programme": PROGRAMME}

# Filled in order. A list means "pick one per user", so thirty users are not
# thirty identical form submissions.
TEXT_INPUTS = {
    "Trainer name": ["Musumbi Denis", "Jane Wanjiru", "Peter Otieno",
                     "Alice Chebet", "Samuel Kiprono"],
    "Institution": "The Rift Valley National Polytechnic",
    "Course": ["ICT Technician", "Computer Operations", "ICT Operator"],
    "Level": ["5", "6"],
    "Number of trainees": ["20", "25", "30", "35"],
    "Class code": ["ICT/T6/2026", "ICT/T5/2026", "CO/L4/2026"],
}

NUMBER_INPUTS = {
    "Term length (weeks)": ["11", "12", "13"],
    "Sessions per week": ["2", "3"],
    "Number of CATs": ["2"],
}

GENERATE_BUTTON = "Generate Learning Plan"

# The unit table is a canvas (glide-data-grid), so there is no row element to
# click - only a position. The select column is ~35px wide and the header ~36px
# tall, both fixed by Streamlit, so the first row's checkbox is reliably here.
GRID_SELECT_X = 17
GRID_ROW_1_Y = 53

# Under 30 users a rerun that takes 2s idle can take far longer, and a timeout
# would be recorded as a failure that is really just queueing. Generous on
# purpose: slowness should show up in the response times, not as errors.
SETTLE_TIMEOUT = 240_000


def _pick(value):
    """One value from a setting that may be a single string or a list."""
    return random.choice(value) if isinstance(value, list) else value


class StreamlitTrainer(PlaywrightUser):
    """One trainer, making one Learning Plan, in one browser session."""

    headless = True
    host = "http://localhost:8501"
    # Think time between journeys. A trainer does not start a second plan the
    # instant the first lands.
    wait_time = between(1, 3)

    # ------------------------------------------------------------- bootstrap #
    async def _pwprep(self) -> None:
        """Launch Chromium ourselves, and launch it once for the whole run.

        Two things are wrong with the plugin's version on this platform. It
        passes --no-zygote and --disable-setuid-sandbox, which are Linux-only
        and leave the browser dying moments after launch (the first new_context
        then fails with "Connection closed while reading from the driver"). And
        it stores the browser on the instance, so every user launches its own
        Chromium - thirty browser processes would make this machine the
        bottleneck rather than the app under test.

        Here the browser lives on the class and each user gets its own
        BrowserContext: a genuinely separate browser session, with its own
        storage and its own Streamlit websocket, which is what the test needs.
        """
        cls = type(self)
        if cls.playwright is None:
            cls.playwright = await async_playwright().start()
        if cls.browser is None:
            cls.browser = await cls.playwright.chromium.launch(
                headless=self.headless,
                channel="chromium",
                args=["--disable-gpu",
                      "--disable-dev-shm-usage",
                      "--disable-accelerated-2d-canvas",
                      "--frame-throttle-fps=10"],
            )

    # ----------------------------------------------------------------- utils #
    async def _settle(self, timeout: int = SETTLE_TIMEOUT) -> None:
        """Wait for Streamlit to finish re-running the script.

        The status widget is mounted while a run is in flight and removed when
        it ends, so 'hidden' (which covers detached) is the finish line. The
        short wait for it to appear first avoids racing past a run that has not
        started drawing yet.
        """
        widget = self.page.locator('[data-testid="stStatusWidget"]')
        try:
            await widget.wait_for(state="visible", timeout=2500)
        except Exception:                 # noqa: BLE001 - rerun already over
            pass
        await widget.wait_for(state="hidden", timeout=timeout)
        await self.page.wait_for_timeout(250)
        await self._check_exception()

    async def _check_exception(self) -> None:
        """A Streamlit traceback on the page fails the step that caused it."""
        exc = self.page.locator('[data-testid="stException"]')
        if await exc.count():
            detail = (await exc.first.inner_text())[:300].replace("\n", " ")
            raise AssertionError(f"stException on the page: {detail}")

    async def _fill(self, label: str, value: str) -> None:
        """Type into the widget labelled `label` and commit it.

        Tab matters: Streamlit commits a text field on blur, so a value typed
        and left focused never reaches the server and never triggers the rerun
        this test is here to measure.
        """
        field = self.page.locator(f'input[aria-label="{label}"]')
        await field.click()
        await field.fill(value)
        await field.press("Tab")
        await self._settle()

    # ------------------------------------------------------------------ task #
    @task
    @pw
    async def make_a_learning_plan(self, page) -> None:
        # `pw` hands the task its page; self.page is the same object, which is
        # what the helpers above use.

        async with event(self, name="01 Load app"):
            await page.goto("/", wait_until="domcontentloaded")
            await self._settle()
            await page.locator('input[aria-label="Programme"]').wait_for()

        # Picking the programme is what fetches both documents and indexes
        # every unit in them - by far the heaviest thing one user asks for.
        async with event(self, name="02 Load documents"):
            box = page.locator('input[aria-label="Programme"]')
            await box.click()
            await box.type(_pick(SELECTBOXES["Programme"]), delay=15)
            await page.wait_for_timeout(600)
            await page.keyboard.press("Enter")
            await self._settle()
            # .first, not the bare locator: when some units in the programme
            # did not pair up, the app also renders the two "unmatched units"
            # tables, and a strict locator then fails on three matches rather
            # than waiting for the one we want.
            await page.locator('[data-testid="stDataFrame"]').first.wait_for()

        # Selecting a row extracts the two units - the second heavy step.
        async with event(self, name="03 Select unit"):
            grid = page.locator('[data-testid="stDataFrame"]').first
            await grid.scroll_into_view_if_needed()
            canvas = grid.locator("canvas").first
            box_ = await canvas.bounding_box()
            if not box_:
                raise AssertionError("unit table has no canvas to click")
            await page.mouse.click(box_["x"] + GRID_SELECT_X,
                                   box_["y"] + GRID_ROW_1_Y)
            await self._settle()
            await page.locator(
                f'[data-testid="stBaseButton-primary"]:has-text("{GENERATE_BUTTON}")'
            ).wait_for()

        # Every field commits on blur, and every commit is a full script rerun -
        # ten of them before the button is even pressed.
        async with event(self, name="04 Fill form"):
            for label, value in TEXT_INPUTS.items():
                await self._fill(label, _pick(value))
            for label, value in NUMBER_INPUTS.items():
                await self._fill(label, _pick(value))

        async with event(self, name="05 Generate plan"):
            await page.locator(
                f'[data-testid="stBaseButton-primary"]:has-text("{GENERATE_BUTTON}")'
            ).click()
            await self._settle()
            await page.get_by_text("Learning Plan ready.").wait_for(timeout=60_000)
            await self._check_exception()


if __name__ == "__main__":
    # One user, start to finish - the dry run. Set headless = False here to
    # watch the browser do it.
    from locust.debug import run_single_user

    run_single_user(StreamlitTrainer)
