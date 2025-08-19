# "Call `triton_backward.run` with JSON: "
# '{"code": "<python module>", "setup": "<python snippet>", '
# '"format": "json|raw"}.\n'
# "Rules:\n"
# "- Define exactly one @triton.jit kernel. In `setup`, run your stub and assign the **LAUNCH** object (has `.asm['ttir']`) to COMPILED_KERNEL (or _compiled_kernel).\n"
# "- Default output is JSON {digest, path}. Use `slice` to page TTIR. Use `format:'raw'` only for tiny graphs.\n"
# "Any for-loops inside the kernel MUST have static bounds."
# "Minimal valid example (\\n escaped):\n"
# '{\n'
# '  "code": "import triton\\nimport triton.language as tl\\nimport torch\\n@triton.jit\\n'
# 'def k(a_ptr,b_ptr,o_ptr):\\n  off=tl.arange(0,4)\\n  tl.store(o_ptr+off, tl.load(a_ptr+off)*tl.load(b_ptr+off))\\n\\n'
# 'def stub(a,b):\\n  o=torch.empty_like(a)\\n  launch = k[(1,1,1)](a,b,o)\\n  return o, launch\\n",\n'
# '  "setup": "import torch\\na=torch.rand(4, device=\\"cuda\\"); b=torch.rand(4, device=\\"cuda\\")\\n_, COMPILED_KERNEL = stub(a,b)",\n'
# '  "format": "json"\n'
# '}\n'


# @property
# def tool_config(self) -> ToolNamespaceConfig:
#     return ToolNamespaceConfig(
#         name=self.get_tool_name(),
#         description="Generate a naive backward TTIR from a compiled Triton kernel.",
#         tools=[
#             ToolDescription.new(
#                 name="run",
#                 description=(
#                     "Compile the provided Triton kernel and return its backward TTIR. "
#                     "Provide Python source for `code` and `setup`. "
#                     "In `setup`, bind COMPILED_KERNEL (or _compiled_kernel) to the **LAUNCH** object."
#                 ),
#                 parameters={
#                     "type": "object",
#                     "properties": {
#                         "code": {
#                             "type": "string",
#                             "description": "Python module defining exactly one @triton.jit kernel.",
#                         },
#                         "setup": {
#                             "type": "string",
#                             "description": "Python snippet that prepares inputs and assigns COMPILED_KERNEL (or _compiled_kernel) to the LAUNCH object.",
#                         },
#                         "format": {
#                             "type": "string",
#                             "enum": ["raw", "json"],
#                             "default": "json",
#                             "description": "Return compact JSON {digest, path} (default) or the raw TTIR text.",
#                         },
#                     },
#                     "required": ["code", "setup"],
#                 },
#             )
#             ,
#             ToolDescription.new(
#                 name="slice",
#                 description="Return a byte-range from the generated TTIR identified by digest.",
#                 parameters={
#                     "type": "object",
#                     "properties": {
#                         "digest": {"type": "string", "description": "Digest (10+ hex chars) returned by run(format=json)."},
#                         "offset": {"type": "integer", "default": 0, "minimum": 0},
#                         "limit":  {"type": "integer", "default": 65536, "minimum": 1},
#                     },
#                     "required": ["digest"],
#                 },
#             )
#         ],
#     )
