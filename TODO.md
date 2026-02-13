# Proof QA Tool - Task List

**Last updated:** 2026-02-13

---

## Completed

- [x] **Rebrand to "Proof by Shawn Hernandez"** — all branding updated, login page logo 50% bigger, tagline with & symbol
- [x] **Add changelog & dev cost calculator to Admin panel** — 12 versions, AI session log, traditional vs AI cost comparison
- [x] **Remove dark mode toggle from navbar** — saved to `dark_mode_reference.md` for future re-implementation
- [x] **Build Timeline X full page** — upload, config (13 formats), style presets, generate, export to DaVinci/FCP/Premiere
- [x] **Build Director X full page** — Claude Vision AI analysis, thoroughness selector, Director Score 0-100, category breakdown, strengths & improvements
- [x] **Navbar reorder** — Photo | Video | Auto Sort | Timeline X BETA | Director X BETA
- [x] **BETA badge styling** — purple #9461F5 with CSS class
- [x] **Remove lock emoji from Sign in button**

## In Progress / Pending

- [ ] **Fix admin email** — change `shawn@aerialcanvas.com` to `shawn.keepitcity@gmail.com` in database.py (`is_admin()`, `is_team_member()`, `create_user()`)
- [ ] **Add Local Path tab to Photo Proof** — third tab for typing/pasting local file path on Mac, no upload needed
- [ ] **Add Local Path tab to Video Proof** — third tab for local video path, same analysis pipeline
- [ ] **Add Local Path tab to Auto Sort (Photo & Video)** — restructure with tabs (Dropbox | Local Path), wire up `local_folder.py` backend for folder scanning, sorting, renaming, `_Sorted` output
- [ ] **Add Local Path tab to Timeline X** — local folder of clips + music file path, use `timeline_x.add_clips_from_folder()` directly
- [ ] **Add Local Path tab to Director X** — local video path for frame extraction + Claude Vision analysis
- [ ] **Update homepage** — stronger opening tagline (replace "Don't suck."), remove "Who's it for" / "Where we're starting" / "Where we're going" sections, add "Ready to stop sucking?" CTA at bottom with "Get started" button routing to Photo Proof

## Future / Backlog

- [ ] Re-implement dark mode (see `dark_mode_reference.md`)
- [ ] Dropbox migration to personal account (see `proof_ownership_migration.md`)
- [ ] GitHub migration (clarify shawnac1 ownership)
- [ ] Streamlit Cloud deployment under personal account
- [ ] Push to remote GitHub regularly

---

## Key Files

- `qa_tool.py` — main app (~16,000+ lines)
- `database.py` — user/stats tracking, admin checks
- `local_folder.py` — local folder processor (exists, needs UI integration)
- `timeline_x.py` — timeline assembly engine
- `timeline_x_analyzer.py` — FFprobe/BPM analysis
- `timeline_x_framework.py` — editorial rules/knowledge base
- `dark_mode_reference.md` — saved dark mode code for later
- `.streamlit/secrets.toml` — OAuth + API keys (NEVER commit)

## Notes

- **Commit after EVERY change** — non-negotiable after losing hours of work on Feb 12
- **Admin email**: shawn.keepitcity@gmail.com
- **Owner**: Shawn Hernandez (NOT Aerial Canvas)
- **Mac-first** — local path features prioritize macOS, PC secondary
- `local_folder.py` backend is complete — just needs UI wiring
