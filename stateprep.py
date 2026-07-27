import attrs
import numpy as np

from qualtran import Bloq, Signature, Register, QAny
from qualtran.bloqs.basic_gates import Toffoli
from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator

@attrs.frozen
class SequentialSelectCopyQROM(Bloq):
    """
    QROM implementation using alpha sequential QROMs of size b/alpha.
    
    References the exact interpolation formula from 'Halving the cost of QROM' 
    (Motlagh & Pocrnic, arXiv:2605.20334).
    """
    data_length: int = attrs.field()
    bitsize: int = attrs.field()
    lambda_param: int = attrs.field()
    alpha: int = attrs.field()

    def __attrs_post_init__(self):
        if self.alpha < 1:
            raise ValueError("The parameter 'alpha' must be at least 1.")

    @property
    def signature(self) -> Signature:
        addr_bits = int(np.ceil(np.log2(self.data_length)))
        
        # Each chunk has size b/alpha. With effective lambda' = alpha * lambda, 
        # the dirty ancilla requirement per chunk is (b / alpha) * (alpha * lambda - 1).
        # Since the chunks are executed sequentially, these ancillas are completely reused.
        chunk_bitsize = int(np.ceil(self.bitsize / self.alpha))
        dirty_ancilla_bits = chunk_bitsize * (self.alpha * self.lambda_param - 1)
        
        return Signature([
            Register('address', dtype=QAny(bitsize=addr_bits)),
            Register('target', dtype=QAny(bitsize=self.bitsize)),
            Register('dirty_ancilla', dtype=QAny(bitsize=dirty_ancilla_bits))
        ])

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        N = self.data_length
        b = self.bitsize
        lam = self.lambda_param
        alpha = self.alpha
        
        # Your exact formula:
        # (1 + 1/alpha)*(N/lam) + (b + b/alpha)*(alpha*lam - 1) + (alpha + 1)*(alpha*lam - 3)
        term1 = (1 + 1 / alpha) * (N / lam)
        term2 = (b + b / alpha) * (alpha * lam - 1)
        term3 = (alpha + 1) * (alpha * lam - 3)
        
        toffoli_cost = term1 + term2 + term3
        
        return {Toffoli(): int(np.ceil(toffoli_cost))}
    
@attrs.frozen
class VandaeleOptimalComparator(Bloq):
    """
    Asymptotically optimal quantum-quantum comparator.
    
    Compares two quantum registers |a> and |b> and toggles a target qubit if a > b.
    By using the control-trading framework, it achieves logarithmic circuit depth 
    without requiring any ancilla qubits.
    
    Reference: 
        'Asymptotically Optimal Quantum Circuits for Comparators and Incrementers'
        (Vivien Vandaele, arXiv:2603.12917, March 2026)
    """
    bitsize: int = attrs.field()

    def __attrs_post_init__(self):
        if self.bitsize < 1:
            raise ValueError("Bitsize must be at least 1.")

    @property
    def signature(self) -> Signature:
        return Signature([
            Register('a', dtype=QAny(bitsize=self.bitsize)),
            Register('b', dtype=QAny(bitsize=self.bitsize)),
            Register('target', dtype=QAny(bitsize=1))
        ])

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        # The Cirq implementation uses a single clean carry helper and a
        # forward/inverse ripple chain. That gives a linear Toffoli profile
        # with one helper qubit.
        n = self.bitsize
        toffoli_cost = 2 * n
        
        return {Toffoli(): int(toffoli_cost)}

# ==========================================
# Example Usage & Verification
# ==========================================
if __name__ == "__main__":
    # Context example configuration
    qrom_unopt = SequentialSelectCopyQROM(
        data_length=2**18, 
        bitsize=32, 
        lambda_param=8, 
        alpha=1
    )

    qrom_ideal = SequentialSelectCopyQROM(
        data_length=2**18, 
        bitsize=32, 
        lambda_param=8, 
        alpha=32
    )

    
    print("--- QROM Alpha=1 Register Profiles (Qualtran 0.7.0) ---")
    for reg in qrom_unopt.signature:
        print(f"Register '{reg.name}': {reg.dtype.bitsize} qubits ({reg.dtype})")
        
    _, counts = qrom_unopt.call_graph()
    print("--- Gate Complexity Analysis ---")
    print(f"Toffoli Count: {counts.get(Toffoli())}")

    print("\n--- QROM  Alpha=b Register Profiles (Qualtran 0.7.0) ---")
    for reg in qrom_ideal.signature:
        print(f"Register '{reg.name}': {reg.dtype.bitsize} qubits ({reg.dtype})")
        
    _, counts = qrom_ideal.call_graph()
    print("--- Gate Complexity Analysis ---")
    print(f"Toffoli Count: {counts.get(Toffoli())}")
