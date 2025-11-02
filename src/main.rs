use std::{num::NonZeroU64, mem::size_of};
use bevy::prelude::*;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use std::sync::mpsc::channel;

const N: usize = 32;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Particle {
    x: f32,
    v: f32,
    a: f32,
}

#[derive(Resource)]
struct SimulationGPU {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout
}

#[derive(Resource)]
struct Buffers {
    input_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    buffer_size: u64
}

#[derive(Component)]
struct ParticleVisual(usize);

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "2D GPU Wave Simulation (Mouse Control)".into(),
                resolution: (800, 400).into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(ClearColor(Color::BLACK))
        .add_systems(Startup, setup_resources)
        .add_systems(Startup, setup_visuals)
        .add_systems(Startup, setup_buffers.after(setup_resources))
        .add_systems(Update, compute_frame)
        .add_systems(Update, apply_user_input.after(compute_frame))
        .run();
}

fn setup_resources(
    mut commands: Commands
) {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        .expect("No adapter found");
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            ..Default::default()
        }
    ))
        .expect("Failed request_device");

    let shader = device.create_shader_module(wgpu::include_wgsl!("simulation.wgsl"));

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("layout"),
        entries: &[
            // input buffer (read-only storage)
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: Some(NonZeroU64::new(size_of::<Particle>() as u64).unwrap()),
                },
                count: None,
            },
            // output buffer (writable storage)
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(NonZeroU64::new(size_of::<Particle>() as u64).unwrap()),
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("entry"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    commands.insert_resource(SimulationGPU {
        device,
        queue,
        pipeline,
        bind_group_layout
    });
}

fn get_initial_conditions() -> Vec<Particle> {
    let particles = vec![Particle { x: 0.0, v: 0.0, a: 0.0 }; N];
    particles
}


fn setup_buffers(mut commands: Commands, sim_gpu: ResMut<SimulationGPU>) {
    let particles= get_initial_conditions();

    let buffer_size = (N * size_of::<Particle>()) as u64;

    let device = &sim_gpu.device;

    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input"),
        contents: bytemuck::cast_slice(&particles),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC |  wgpu::BufferUsages::COPY_DST,
    });

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // To read back to the CPU and render the visualization
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    commands.insert_resource(Buffers {
        input_buffer,
        output_buffer,
        readback_buffer,
        buffer_size
    });
}

fn setup_visuals(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, mut materials: ResMut<Assets<ColorMaterial>>) {
    commands.spawn(Camera2d);

    let spacing = 20.0;
    for i in 0..N {
        let x = (i as f32 - N as f32 / 2.0) * spacing;
        commands.spawn((
            Mesh2d(meshes.add(Circle::new(5.0))),
            MeshMaterial2d(materials.add(ColorMaterial::from(Color::WHITE))),
            Transform::from_xyz(x, 0.0, 0.0),
            ParticleVisual(i),
        ));
    }
}

fn compute_frame(
    sim_gpu: Res<SimulationGPU>,
    buffers: ResMut<Buffers>,
    mut query: Query<(&ParticleVisual, &mut Transform)>
) {
    let device = &sim_gpu.device;
    let queue = &sim_gpu.queue;

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind group"),
        layout: &sim_gpu.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffers.input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffers.output_buffer.as_entire_binding(),
            },
        ],
    });

    // Encoder registers commands but does not execute them yet.
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("compute encoder") });

    // Dispatch call to the compute shader
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        cpass.set_pipeline(&sim_gpu.pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(((N + 63) / 64) as u32, 1, 1);
    }

    // Copy output to readback buffer
    encoder.copy_buffer_to_buffer(&buffers.output_buffer, 0, &buffers.readback_buffer, 0, buffers.buffer_size);
    // Copy output back to input for the next frame
    encoder.copy_buffer_to_buffer(&buffers.output_buffer, 0, &buffers.input_buffer, 0, buffers.buffer_size);

    // Finish the encoded buffer
    let command_buffer = encoder.finish();

    // Submit the command buffer to the GPU
    queue.submit(Some(command_buffer));

    // Wait for GPU to finish
    device.poll(wgpu::PollType::wait()).unwrap();

    // Readback to the CPU. TODO: Make it async for better performance.
    let slice = buffers.readback_buffer.slice(..);
    let (tx, rx) = channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        tx.send(res).ok();
    });

    // wait for mapping to complete
    device.poll(wgpu::PollType::wait()).unwrap();
    rx.recv().expect("map_async channel recv failed").expect("map_async failed");

    // read mapped data
    let data = slice.get_mapped_range();
    let updated: &[Particle] = bytemuck::cast_slice(&data);

    // copy results to CPU-side particles
    let particles = Vec::from(updated);

    // unmap the buffer so the GPU can reuse it next frame
    drop(data); // drop data to safely unmap the buffer
    buffers.readback_buffer.unmap();


    // update visuals
    for (ParticleVisual(i), mut transform) in &mut query {
        // The last one is exempt
        if *i != N - 1 {
            transform.translation.y = particles[*i].x * 100.0;
        }

    }
}


fn apply_user_input(
    windows: Query<&Window>,
    sim_gpu: Res<SimulationGPU>,
    buffers: Res<Buffers>,
    mut query: Query<(&ParticleVisual, &mut Transform)>
) {
    let window = windows.single().unwrap();
    let Some(cursor_pos) = window.cursor_position() else { return };

    let height = window.height();
    let norm_y = cursor_pos.y / height;
    let sim_x = (0.5 - norm_y) * 4.0;

    for (visual, mut transform) in &mut query {
        if visual.0 == N - 1 {
            transform.translation.y = sim_x * 100.0;
            break;
        }
    }

    let last_particle_state = Particle {
        x: sim_x,
        v: 0.0,
        a: 0.0
    };

    let offset = (N - 1) * size_of::<Particle>();
    let bytes = bytemuck::bytes_of(&last_particle_state);

    // Write the new state directly to the input buffer for the next frame
    sim_gpu.queue.write_buffer(&buffers.input_buffer, offset as u64, bytes);
}