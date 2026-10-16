"""Shrink falcon_ocr_kv_token.onnx for the browser.

MatMul weights -> 4-bit or 8-bit MatMulNBits (fp32 activations, WebGPU-supported).
Optionally (--emb-fp16) the token embedding table -> fp16 + Cast: saves 100 MB but the
fp16 Gather needs the WebGPU `shader-f16` feature, which some GPUs/drivers lack.

Usage: python3 -I tools/quantize.py models/falcon-ocr/falcon_ocr_kv_token.onnx out.onnx [bits] [block] [--emb-fp16]
"""
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer, DefaultWeightOnlyQuantConfig

args = [a for a in sys.argv[1:] if not a.startswith('--')]
emb_fp16 = '--emb-fp16' in sys.argv
src, dst = args[0], args[1]
bits = int(args[2]) if len(args) > 2 else 4
block = int(args[3]) if len(args) > 3 else 32

model = onnx.load(src)
cfg = DefaultWeightOnlyQuantConfig(block_size=block, is_symmetric=True, bits=bits, accuracy_level=0)
q = MatMulNBitsQuantizer(model, algo_config=cfg)
q.process()
model = q.model.model

g = model.graph
emb = None if not emb_fp16 else next(t for t in g.initializer if t.name == 'base.tok_embeddings.weight')
if emb is not None:
    arr = numpy_helper.to_array(emb).astype(np.float16)
    g.initializer.remove(emb)
    g.initializer.append(numpy_helper.from_array(arr, 'base.tok_embeddings.weight_fp16'))
for i, n in enumerate(list(g.node) if emb is not None else []):
    if n.op_type == 'Gather' and n.input[0] == 'base.tok_embeddings.weight':
        out = n.output[0]
        n.input[0] = 'base.tok_embeddings.weight_fp16'
        n.output[0] = out + '_fp16'
        g.node.insert(i + 1, helper.make_node('Cast', [out + '_fp16'], [out], to=onnx.TensorProto.FLOAT))

onnx.save(model, dst)
print('ops', sorted({n.op_type for n in g.node}))
