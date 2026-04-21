import os
import tempfile

from ignite.handlers import ModelCheckpoint as IgniteModelCheckpoint


class ModelCheckpoint(IgniteModelCheckpoint):
    """Windows-safe checkpoint handler for the old Ignite version in this repo."""

    def _save(self, obj, path):
        if not self._atomic:
            self._internal_save(obj, path)
            return

        tmp = tempfile.NamedTemporaryFile(delete=False, dir=self._dirname)
        try:
            self._internal_save(obj, tmp.file)
        except BaseException:
            tmp.close()
            os.remove(tmp.name)
            raise
        else:
            tmp.close()
            os.replace(tmp.name, path)
