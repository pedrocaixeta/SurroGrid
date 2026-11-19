"""Gemeinsame Helfer: Streaming-Parser fuer den MaStR-Gesamtdatenexport (UTF-16 XML in ZIP)."""
import codecs
import io
import re
import urllib.request
import zipfile

CHUNK = 32 * 1024 * 1024  # bytes (komprimiert entpackt)


class HttpFile(io.RawIOBase):
    """Dateiobjekt ueber HTTP-Range-Requests (nur zum Testen ohne lokalen Download)."""

    def __init__(self, url):
        self.url = url
        self.size = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD")).headers["Content-Length"])
        self.pos = 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        n = len(b)
        if n == 0 or self.pos >= self.size:
            return 0
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        data = urllib.request.urlopen(req).read()
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def open_zip(src):
    if src.startswith("http"):
        return zipfile.ZipFile(io.BufferedReader(HttpFile(src), buffer_size=1 << 20))
    return zipfile.ZipFile(src)


def iter_records(z, member, tag, max_records=None):
    """Liefert die Roh-Strings (<tag>...</tag>) jedes Datensatzes in der XML-Datei."""
    end = f"</{tag}>"
    start = f"<{tag}>"
    dec = codecs.getincrementaldecoder("utf-16")()
    buf = ""
    n = 0
    with z.open(member) as fh:
        while True:
            raw = fh.read(CHUNK)
            if not raw:
                break
            buf += dec.decode(raw)
            pos = 0
            while True:
                e = buf.find(end, pos)
                if e < 0:
                    break
                s = buf.rfind(start, pos, e)
                yield buf[s : e + len(end)]
                n += 1
                if max_records and n >= max_records:
                    return
                pos = e + len(end)
            buf = buf[pos:]


def field(rec, name):
    m = re.search(rf"<{name}>(.*?)</{name}>", rec, re.S)
    return m.group(1) if m else None

