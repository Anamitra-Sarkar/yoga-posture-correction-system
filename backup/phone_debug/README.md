# phone_debug — test the web app on a real Android phone over USB

Needs: `adb` (installed), USB debugging on the phone, Chrome open on the phone, `adb forward tcp:9222 localabstract:chrome_devtools_remote`.
* `phone_cdp.py` — minimal Chrome-DevTools-Protocol client. It only ever touches a tab it creates; it never reads or changes the owner's other tabs.
* `run_app_on_phone.py <url> [seconds]` — opens the app in a NEW tab on the phone, grants the camera to that tab, presses Start, and logs every WebGL context request, console warning, the engine in use and the number of pose results per interval. **The tab is visible on the phone and uses the front camera**: tell the phone's owner before running it. For Vercel preview URLs use a share link (`get_access_to_vercel_url`).
Findings from the first session: `docs/BENCHMARKS.md` section 11.
