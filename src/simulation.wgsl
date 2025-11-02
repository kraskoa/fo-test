const dt: f32 = 0.05;
const k: f32 = 0.03;

struct Particle {
    x: f32,
    v: f32,
    a: f32,
};

@group(0) @binding(0)
var<storage, read> input: array<Particle>;

@group(0) @binding(1)
var<storage, read_write> output: array<Particle>;

fn updateParticle(index: u32, array_length: u32) {
    // Skip boundaries
    if (index == 0u || index >= array_length - 1u) {
        return;
    }

    let prev = input[index - 1u];
    let next = input[index + 1u];
    let curr = input[index];

    var updated = curr;

    // Velocity Verlet integration
    updated.v += 0.5 * curr.a * dt;
    updated.x += updated.v * dt;
    updated.a = k * (prev.x - 2.0 * curr.x + next.x);
    updated.v += 0.5 * updated.a * dt;

    output[index] = updated;
}

@compute @workgroup_size(64)
fn entry(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let array_length = arrayLength(&input);
    if (index >= array_length) {
        return;
    }

    updateParticle(index, array_length);
}