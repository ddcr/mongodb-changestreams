25/09/2024:
- Correction of bug: "prefect start server" does not work
  See pull request 21:
    Fixes prefect server start for Windows machines #15103
  Download patch as:
    wget https://github.com/PrefectHQ/prefect/commit/df293ab3a15933c70ae8c06d0974989d22c5035b.patch

- Resolve error "Failed to send telemetry":
  See https://github.com/PrefectHQ/prefect/issues/11326:
    prefect config set PREFECT_SERVER_ANALYTICS_ENABLED=False