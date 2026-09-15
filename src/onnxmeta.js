// Minimal protobuf walker that pulls metadata_props and the first input shape out of an ONNX
// ModelProto without decoding the (huge) weights. Enough to auto-configure Ultralytics exports.

function* fields(buf, start, end) {
  let p = start;
  while (p < end) {
    let tag = 0;
    let shift = 0;
    let b;
    do {
      b = buf[p++];
      tag |= (b & 0x7f) << shift;
      shift += 7;
    } while (b & 0x80);
    const field = tag >>> 3;
    const wire = tag & 7;
    if (wire === 0) {
      let v = 0n;
      shift = 0n;
      do {
        b = buf[p++];
        v |= BigInt(b & 0x7f) << shift;
        shift += 7n;
      } while (b & 0x80);
      yield { field, wire, value: v };
    } else if (wire === 2) {
      let len = 0;
      shift = 0;
      do {
        b = buf[p++];
        len |= (b & 0x7f) << shift;
        shift += 7;
      } while (b & 0x80);
      yield { field, wire, start: p, end: p + len };
      p += len;
    } else if (wire === 1) {
      p += 8;
    } else if (wire === 5) {
      p += 4;
    } else {
      throw new Error('Unsupported protobuf wire type ' + wire);
    }
  }
}

const utf8 = new TextDecoder();
const str = (buf, f) => utf8.decode(buf.subarray(f.start, f.end));

/** @returns {{[key: string]: string, inputName?: string, inputShape?: number[]}} */
export function readOnnxMeta(arrayBuffer) {
  const buf = new Uint8Array(arrayBuffer);
  const meta = {};
  for (const f of fields(buf, 0, buf.length)) {
    if (f.field === 14 && f.wire === 2) {
      // StringStringEntryProto { key = 1, value = 2 }
      let key = '';
      let value = '';
      for (const g of fields(buf, f.start, f.end)) {
        if (g.field === 1) key = str(buf, g);
        else if (g.field === 2) value = str(buf, g);
      }
      meta[key] = value;
    } else if (f.field === 7 && f.wire === 2 && !meta.inputShape) {
      // GraphProto.input = 11 → ValueInfoProto { name = 1, type = 2 }
      for (const g of fields(buf, f.start, f.end)) {
        if (g.field !== 11) continue;
        for (const v of fields(buf, g.start, g.end)) {
          if (v.field === 1) meta.inputName = str(buf, v);
          else if (v.field === 2) meta.inputShape = readShape(buf, v);
        }
        break;
      }
    }
  }
  return meta;
}

// TypeProto.tensor_type = 1 → Tensor { shape = 2 } → TensorShapeProto { dim = 1 } → Dimension { dim_value = 1 }
function readShape(buf, typeField) {
  const dims = [];
  for (const t of fields(buf, typeField.start, typeField.end)) {
    if (t.field !== 1) continue;
    for (const s of fields(buf, t.start, t.end)) {
      if (s.field !== 2) continue;
      for (const d of fields(buf, s.start, s.end)) {
        if (d.field !== 1) continue;
        let val = 0;
        for (const x of fields(buf, d.start, d.end)) if (x.field === 1 && x.wire === 0) val = Number(x.value);
        dims.push(val);
      }
    }
  }
  return dims;
}
