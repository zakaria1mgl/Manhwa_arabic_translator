import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app.pipeline
import app.batch.chapter
import app.quality_control.qa
import app.ui.editor_state
import app.translation.sfx
import app.context.chapter_context

# gradio UI module: only importable when gradio is installed
try:
    import app.ui.gradio_app
    print('GRADIO_UI_IMPORT_OK')
except ModuleNotFoundError as e:
    print(f'gradio UI skipped locally: {e.name} not installed')

print('PIPELINE_IMPORT_OK')
