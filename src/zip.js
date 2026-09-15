// Tiny ZIP writer (STORE method, no compression) – enough to bundle PNGs / JSON for download.
const CRC_TABLE = new Int32Array(256);
for (let n = 0; n < 256; n++) {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  CRC_TABLE[n] = c;
}

function crc32(bytes) {
  let c = -1;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ -1) >>> 0;
}

function dosDateTime(d = new Date()) {
  const time = (d.getHours() << 11) | (d.getMinutes() << 5) | (d.getSeconds() >> 1);
  const date = ((d.getFullYear() - 1980) << 9) | ((d.getMonth() + 1) << 5) | d.getDate();
  return { time, date };
}

export class ZipWriter {
  constructor() {
    this.entries = [];
  }

  /** @param {string} name  @param {Uint8Array|ArrayBuffer|Blob|string} data */
  async add(name, data) {
    let bytes;
    if (typeof data === 'string') bytes = new TextEncoder().encode(data);
    else if (data instanceof Blob) bytes = new Uint8Array(await data.arrayBuffer());
    else if (data instanceof ArrayBuffer) bytes = new Uint8Array(data);
    else bytes = data;
    this.entries.push({ name: new TextEncoder().encode(name), bytes, crc: crc32(bytes) });
  }

  blob() {
    const { time, date } = dosDateTime();
    const parts = [];
    const central = [];
    let offset = 0;
    for (const e of this.entries) {
      const local = new DataView(new ArrayBuffer(30));
      local.setUint32(0, 0x04034b50, true);
      local.setUint16(4, 20, true); // version needed
      local.setUint16(6, 0x0800, true); // utf-8 names
      local.setUint16(8, 0, true); // STORE
      local.setUint16(10, time, true);
      local.setUint16(12, date, true);
      local.setUint32(14, e.crc, true);
      local.setUint32(18, e.bytes.length, true);
      local.setUint32(22, e.bytes.length, true);
      local.setUint16(26, e.name.length, true);
      local.setUint16(28, 0, true);
      parts.push(local.buffer, e.name, e.bytes);

      const cd = new DataView(new ArrayBuffer(46));
      cd.setUint32(0, 0x02014b50, true);
      cd.setUint16(4, 20, true);
      cd.setUint16(6, 20, true);
      cd.setUint16(8, 0x0800, true);
      cd.setUint16(10, 0, true);
      cd.setUint16(12, time, true);
      cd.setUint16(14, date, true);
      cd.setUint32(16, e.crc, true);
      cd.setUint32(20, e.bytes.length, true);
      cd.setUint32(24, e.bytes.length, true);
      cd.setUint16(28, e.name.length, true);
      cd.setUint32(42, offset, true);
      central.push(cd.buffer, e.name);
      offset += 30 + e.name.length + e.bytes.length;
    }
    const cdSize = central.reduce((n, c) => n + (c.byteLength ?? c.length), 0);
    const end = new DataView(new ArrayBuffer(22));
    end.setUint32(0, 0x06054b50, true);
    end.setUint16(8, this.entries.length, true);
    end.setUint16(10, this.entries.length, true);
    end.setUint32(12, cdSize, true);
    end.setUint32(16, offset, true);
    return new Blob([...parts, ...central, end.buffer], { type: 'application/zip' });
  }
}
