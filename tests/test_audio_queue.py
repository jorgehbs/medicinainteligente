import asyncio
import sys
import types
import unittest


def _ensure_stub_modules():
    if 'fastapi' not in sys.modules:
        fastapi_module = types.ModuleType('fastapi')

        class DummyApp:
            def __init__(self, *args, **kwargs):
                pass

            def mount(self, *args, **kwargs):
                pass

            def get(self, *args, **kwargs):
                def decorator(func):
                    return func

                return decorator

            def websocket(self, *args, **kwargs):
                def decorator(func):
                    return func

                return decorator

            def on_event(self, *args, **kwargs):
                def decorator(func):
                    return func

                return decorator

        class DummyWebSocket:
            pass

        class DummyHTTPException(Exception):
            pass

        fastapi_module.FastAPI = DummyApp
        fastapi_module.WebSocket = DummyWebSocket
        fastapi_module.WebSocketDisconnect = Exception
        fastapi_module.HTTPException = DummyHTTPException
        sys.modules['fastapi'] = fastapi_module

        staticfiles_module = types.ModuleType('fastapi.staticfiles')

        class StaticFiles:
            def __init__(self, *args, **kwargs):
                pass

        staticfiles_module.StaticFiles = StaticFiles
        sys.modules['fastapi.staticfiles'] = staticfiles_module

        responses_module = types.ModuleType('fastapi.responses')

        class HTMLResponse(str):
            def __new__(cls, content, *args, **kwargs):
                return str.__new__(cls, content)

        responses_module.HTMLResponse = HTMLResponse
        sys.modules['fastapi.responses'] = responses_module

    if 'uvicorn' not in sys.modules:
        uvicorn_module = types.ModuleType('uvicorn')

        def run(*args, **kwargs):
            raise RuntimeError('uvicorn.run should not be invoked during tests')

        uvicorn_module.run = run
        sys.modules['uvicorn'] = uvicorn_module

    if 'openai' not in sys.modules:
        openai_module = types.ModuleType('openai')

        class _DummyTranscriptions:
            def create(self, *args, **kwargs):
                return types.SimpleNamespace(text="")

        class _DummyAudio:
            def __init__(self):
                self.transcriptions = _DummyTranscriptions()

        class DummyClient:
            def __init__(self, *args, **kwargs):
                self.audio = _DummyAudio()

        openai_module.OpenAI = DummyClient
        sys.modules['openai'] = openai_module

    if 'pydub' not in sys.modules:
        pydub_module = types.ModuleType('pydub')

        class AudioSegment:
            pass

        pydub_module.AudioSegment = AudioSegment
        sys.modules['pydub'] = pydub_module

    if 'spacy' not in sys.modules:
        spacy_module = types.ModuleType('spacy')

        def load(*args, **kwargs):
            raise OSError('spaCy models unavailable in test environment')

        spacy_module.load = load
        sys.modules['spacy'] = spacy_module

        lang_pt_module = types.ModuleType('spacy.lang.pt')

        class Portuguese:
            def __init__(self, *args, **kwargs):
                pass

            def add_pipe(self, *args, **kwargs):
                return None

        lang_pt_module.Portuguese = Portuguese
        sys.modules['spacy.lang.pt'] = lang_pt_module

        negspacy_module = types.ModuleType('negspacy')
        negation_module = types.ModuleType('negspacy.negation')

        class Negex:
            def __init__(self, *args, **kwargs):
                pass

        negation_module.Negex = Negex
        negspacy_module.negation = negation_module
        sys.modules['negspacy'] = negspacy_module
        sys.modules['negspacy.negation'] = negation_module

    if 'pydantic' not in sys.modules:
        pydantic_module = types.ModuleType('pydantic')

        class FieldInfo:
            def __init__(self, default=None, default_factory=None, **kwargs):
                self.default = default
                self.default_factory = default_factory

        def Field(*, default=None, default_factory=None, **kwargs):
            return FieldInfo(default=default, default_factory=default_factory)

        class BaseModelMeta(type):
            def __new__(mcls, name, bases, namespace):
                annotations = dict(namespace.get('__annotations__', {}))
                field_defaults = {}

                for field_name in annotations:
                    value = namespace.get(field_name, FieldInfo())

                    if isinstance(value, FieldInfo):
                        field_defaults[field_name] = value
                        namespace[field_name] = None
                    else:
                        field_defaults[field_name] = FieldInfo(default=value)

                namespace['__field_defaults__'] = field_defaults
                return super().__new__(mcls, name, bases, namespace)

        class BaseModel(metaclass=BaseModelMeta):
            def __init__(self, **data):
                for field_name, field_info in self.__field_defaults__.items():
                    if field_name in data:
                        value = data[field_name]
                    elif field_info.default_factory is not None:
                        value = field_info.default_factory()
                    else:
                        value = field_info.default
                    setattr(self, field_name, value)

            def model_dump(self):
                return self.__dict__.copy()

        pydantic_module.BaseModel = BaseModel
        pydantic_module.Field = Field
        sys.modules['pydantic'] = pydantic_module


_ensure_stub_modules()

import main


class ProcessAudioQueueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.encounter_id = "test-encounter"
        self.sent_updates = []

        # Prepare shared state
        main.TRANSCRIPTS[self.encounter_id] = ""
        main.AUDIO_QUEUES[self.encounter_id] = asyncio.Queue()

        async def fake_transcribe_audio_chunk(chunk: bytes) -> str:
            await asyncio.sleep(0)
            return chunk.decode("utf-8")

        async def fake_broadcast_transcript_update(encounter_id: str, transcript_text: str):
            self.sent_updates.append((encounter_id, transcript_text))

        # Patch async helpers
        self._original_transcribe = main.transcribe_audio_chunk
        self._original_broadcast = main.broadcast_transcript_update
        main.transcribe_audio_chunk = fake_transcribe_audio_chunk
        main.broadcast_transcript_update = fake_broadcast_transcript_update

    async def asyncTearDown(self):
        # Restore patched helpers
        main.transcribe_audio_chunk = self._original_transcribe
        main.broadcast_transcript_update = self._original_broadcast

        # Cleanup shared state
        main.TRANSCRIPTS.pop(self.encounter_id, None)
        queue = main.AUDIO_QUEUES.pop(self.encounter_id, None)
        if queue is not None:
            while not queue.empty():
                queue.get_nowait()
                queue.task_done()

    async def test_process_audio_queue_in_order(self):
        worker = asyncio.create_task(main.process_audio_queue(self.encounter_id))

        try:
            await main.AUDIO_QUEUES[self.encounter_id].put(b"Ola")
            await main.AUDIO_QUEUES[self.encounter_id].put(b" Jorge")

            await asyncio.wait_for(main.AUDIO_QUEUES[self.encounter_id].join(), timeout=1)

            self.assertEqual(main.TRANSCRIPTS[self.encounter_id], "Ola Jorge")
            self.assertEqual(
                self.sent_updates,
                [
                    (self.encounter_id, "Ola"),
                    (self.encounter_id, "Ola Jorge"),
                ],
            )
        finally:
            worker.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await worker


if __name__ == "__main__":
    unittest.main()
