# Makes the small model the test uses: input "x" float32 [1,4], output "y" = 2x.  python make_test_model.py out.mlpackage
import sys, torch, coremltools as ct
class M(torch.nn.Module):
    def forward(self, x): return x * 2
ex = torch.export.export(M().eval(), (torch.rand(1, 4),)).run_decompositions({})
ct.convert(ex, inputs=[ct.TensorType(name="x", shape=(1, 4))], outputs=[ct.TensorType(name="y")],
           minimum_deployment_target=ct.target.macOS14, compute_precision=ct.precision.FLOAT32).save(sys.argv[1])
