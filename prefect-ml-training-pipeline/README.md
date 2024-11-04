Prefect Deployment
==================

```bash
prefect deployment build train_yoloflow.py:yolo_workflow -n 'ml_workflow_yolo' -a --tag yolo
```

Prefect Logging (optional)
==========================

By default, Prefect outputs its logs to the console and sends them to the Prefect 
API (either Prefect Server or Prefect Cloud). This means that all will be logged
in the output of `Prefect Agent`

Create custom logging.yaml and then run
```bash
prefect config set PREFECT_LOGGING_SETTINGS_PATH=/path/to/your/logging.yml
```
The location can be `$HOME/.prefect/logging.yaml`
