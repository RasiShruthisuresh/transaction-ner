# Project rules — Transaction NER assignment

## Context discipline
- Never print or dump raw dataset rows into the terminal/conversation. All data
  inspection goes through scripts that report `.shape`, `.head()`, value counts,
  or other summaries — never full file contents. Save full outputs to files if
  the user needs to inspect them, don't paste them into the response.
- Redirect any training/evaluation run's stdout to a log file under `logs/`
  (e.g. `logs/phase3_train.log`). Only read the last ~50 lines of a log unless
  explicitly asked to see more.
- For pure investigation tasks (checking a file's schema, searching for a
  pattern, reading multiple files to understand something) use a subagent and
  report back only the summary — don't pull the raw exploration into the main
  thread.
- Keep responses terse. Report what was done, what was verified, and what's
  next. Skip preamble and restating the task back to the user.

## Session hygiene
- After a phase is implemented, tested, and committed+pushed, stop and tell
  the user it's safe to run `/clear`. Before stopping, write/update
  `PROGRESS.md` at the repo root with: which phase just finished, what was
  decided and why, and what the next phase is. That file is how the next
  session picks up context — don't rely on the user re-pasting the original
  spec.
- At the start of a new session, read `PROGRESS.md` and this file first, then
  report what was found and ask the user to confirm before starting the next
  phase.

## Hard rules — never relax these
- Never call the real submission API (`POST /submit`) without telling the user
  exactly what's about to be submitted and getting their explicit go-ahead
  first. There are 20 attempts total for the whole assignment.
- Don't commit or push unverified/broken work. Test each phase before
  committing.

## Environment quirks (this machine)
- `git push` may fail with a schannel `CRYPT_E_NO_REVOCATION_CHECK` /
  connection-timeout error (Avast's HTTPS interception appears to block the
  OCSP revocation lookup). Workaround: prefix with
  `GIT_SSL_NO_REVOKE=true git push`. Do not change git config to work around
  this permanently — use the env var per-call.
- `uv pip install` needs `--system-certs` for the same reason (TLS
  interception breaks uv's bundled CA bundle).
- PowerShell is not available in this environment; use the Bash tool
  (Git Bash / POSIX sh) for shell commands.

## Repo layout
See README.md for the up-to-date module/file layout and phase checklist.
