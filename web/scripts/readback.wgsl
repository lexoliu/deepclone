@group(0) @binding(0) var<storage, read_write> values: array<u32>;

// A fixed, small workload measures round-trip overhead rather than throughput.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) index: vec3<u32>) {
    if index.x < arrayLength(&values) {
        values[index.x] += 1u;
    }
}
