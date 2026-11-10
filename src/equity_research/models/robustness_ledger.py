"""Durable, single-writer accounting for a finite set of fitting tasks."""
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


class FitLedger:
    """Reserve before fit; unfinished/failed attempts are never retried implicitly."""
    def __init__(self, directory, manifest, tasks, budget, *, resume=False):
        self.directory = Path(directory)
        self.path = self.directory/'fit_ledger.json'
        self.lock = self.directory/'.writer.lock'
        self.tasks = {task['key']: task for task in tasks}
        if len(self.tasks) != len(tasks):
            raise ValueError('Duplicate task key')
        if self.directory.exists() and not resume:
            raise ValueError('Existing experiment; explicit resume required')
        if resume and not self.path.exists():
            raise ValueError('No ledger to resume')
        self.directory.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            os.write(self.fd, str(os.getpid()).encode()); os.fsync(self.fd)
            if resume:
                self.state = json.loads(self.path.read_text(encoding='utf-8'))
                if self.state['manifest'] != manifest or self.state['tasks'] != self.tasks or self.state['budget'] != budget:
                    raise ValueError('Resume manifest, tasks or budget changed')
                self.verify()
            else:
                self.state = dict(manifest=manifest, tasks=self.tasks, budget=budget, entries={})
                self.save()
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self.fd is not None:
            os.close(self.fd); self.fd = None
            self.lock.unlink()

    def save(self):
        atomic_json(self.path, self.state)

    def artifact_path(self, relative):
        path = (self.directory/relative).resolve()
        if not path.is_relative_to(self.directory.resolve()):
            raise ValueError('Artifact outside run directory')
        return path

    @property
    def used(self):
        return sum(entry['kind'] == 'fit' for entry in self.state['entries'].values())

    def verify(self):
        for key, entry in self.state['entries'].items():
            if key not in self.tasks or entry['kind'] != self.tasks[key]['kind']:
                raise ValueError('Ledger task mismatch')
            if entry['status'] not in {'completed', 'failed', 'running'}:
                raise ValueError('Unknown ledger status')
            if entry['status'] != 'completed':
                raise ValueError('Interrupted or failed task requires review; no automatic retry')
            if not entry.get('artifacts'):
                raise ValueError('Completed task has no artifacts')
            for name, digest in entry['artifacts'].items():
                if sha(self.artifact_path(name)) != digest:
                    raise ValueError('Completed artifact changed')
        if self.used > self.state['budget']:
            raise ValueError('Fit budget exceeded')

    def completed(self, key):
        entry = self.state['entries'].get(key)
        if entry is None:
            return False
        if entry['status'] != 'completed':
            raise ValueError('Task already reserved or failed; no retry')
        for name, digest in entry['artifacts'].items():
            if sha(self.artifact_path(name)) != digest:
                raise ValueError('Completed artifact changed')
        return True

    def perform(self, task, operation):
        """operation writes artifacts and returns their run-relative paths."""
        key = task['key']
        if self.tasks.get(key) != task:
            raise ValueError('Task not in frozen schedule')
        if self.completed(key):
            return False
        if task['kind'] == 'fit' and self.used >= self.state['budget']:
            raise ValueError('Fit budget exhausted')
        self.state['entries'][key] = dict(kind=task['kind'], status='running')
        self.save()  # Durable reservation precedes the only operation allowed to fit.
        try:
            files = operation(task)
            if not files:
                raise ValueError('No task artifacts')
            digests = {name: sha(self.artifact_path(name)) for name in files}
            self.state['entries'][key].update(status='completed', artifacts=digests)
            self.save()
        except BaseException as error:
            self.state['entries'][key].update(status='failed', error_type=type(error).__name__)
            self.save()
            raise
        return True
