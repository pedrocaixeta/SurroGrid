import sys, zipfile, io, urllib.request

URL = sys.argv[1]

class HttpFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        req = urllib.request.Request(url, method="HEAD")
        self.size = int(urllib.request.urlopen(req).headers["Content-Length"])
        self.pos = 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos
    def readinto(self, b):
        n = len(b)
        if n == 0 or self.pos >= self.size: return 0
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        data = urllib.request.urlopen(req).read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)

f = io.BufferedReader(HttpFile(URL), buffer_size=1 << 20)
z = zipfile.ZipFile(f)
for i in z.infolist():
    print(f"{i.file_size:>14,}  {i.compress_size:>12,}  {i.filename}")

