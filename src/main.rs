use vulkano::buffer::cpu_pool::CpuBufferPool;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState, SubpassContents};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor::PipelineLayoutAbstract;
use vulkano::device::{Device, DeviceExtensions, Queue};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass};
use vulkano::image::{Dimensions, ImageUsage, StorageImage, SwapchainImage};
use vulkano::instance::Instance;
use vulkano::instance::PhysicalDevice;
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::swapchain;
use vulkano::swapchain::{
    AcquireError, ColorSpace, FullscreenExclusive, PresentMode, SurfaceTransform, Swapchain,
    SwapchainCreationError,
};
use vulkano::sync;
use vulkano::sync::{FlushError, GpuFuture};

use vulkano_win::VkSurfaceBuild;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, VirtualKeyCode};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};
use winit_input_helper::WinitInputHelper;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::sync::Arc;
use std::time::SystemTime;
use vecmath::{vec3_add, vec3_cross, vec3_scale};

static NUMBER_OF_SUBDIVISIONS: usize = 100000;
static RENDER_BUFFER_COUNT: u32 = 3;
const FIBONACHI_LENGTH: usize = 20;

fn main() {
    // #region Initilisation
    let instance = Instance::new(None, &vulkano_win::required_extensions(), None).unwrap();
    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();

    let mut input = WinitInputHelper::new();

    let event_loop = EventLoop::new();

    let surface = WindowBuilder::new()
        .with_title("Vulkano Test")
        .with_inner_size(LogicalSize {
            width: 720,
            height: 720,
        })
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    surface.window().set_cursor_grab(true).unwrap();
    surface.window().set_cursor_visible(false);

    let queue_family = physical
        .queue_families()
        .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
        .unwrap();

    let device_ext = DeviceExtensions {
        khr_swapchain: true,
        khr_storage_buffer_storage_class: true,
        ..DeviceExtensions::none()
    };
    let (device, mut queues) = Device::new(
        physical,
        physical.supported_features(),
        &device_ext,
        [(queue_family, 0.5)].iter().cloned(),
    )
    .unwrap();

    let queue = queues.next().unwrap();
    // #endregion

    // #region Image setup
    let dimensions: [u32; 2] = surface.window().inner_size().into();

    let (mut swapchain, output_images) = {
        let capabilities = surface.capabilities(physical).unwrap();
        let alpha = capabilities
            .supported_composite_alpha
            .iter()
            .next()
            .unwrap();
        let format = capabilities.supported_formats[0].0;
        Swapchain::new(
            device.clone(),
            surface.clone(),
            capabilities.min_image_count,
            format,
            dimensions,
            1,
            ImageUsage::color_attachment(),
            &queue,
            SurfaceTransform::Identity,
            alpha,
            PresentMode::Fifo,
            FullscreenExclusive::Default,
            true,
            ColorSpace::SrgbNonLinear,
        )
        .unwrap()
    };
    // #endregion

    // #region Buffer setup
    #[derive(Default, Debug, Clone)]
    struct Vertex {
        position: [f32; 2],
    }

    let vertex_buffer = {
        vulkano::impl_vertex!(Vertex, position);

        let vertex_array = [
            Vertex {
                position: [1.0, 1.0],
            },
            Vertex {
                position: [-1.0, 1.0],
            },
            Vertex {
                position: [1.0, -1.0],
            },
            Vertex {
                position: [-1.0, -1.0],
            },
        ]
        .iter()
        .cloned();

        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, vertex_array)
            .unwrap()
    };

    let graphics_uniform_buffer =
        CpuBufferPool::<graphics_fs::ty::Uniforms>::new(device.clone(), BufferUsage::all());
    let post_uniform_buffer =
        CpuBufferPool::<post_fs::ty::Uniforms>::new(device.clone(), BufferUsage::all());

    let data_len;
    let data_buffer = {
        // let root_node_bytes = [26u8, 201, 137, 0];
        // let root_node = u32::from_be_bytes(root_node_bytes);
        let mut rng: SmallRng = SeedableRng::seed_from_u64(0);
        let mut nodes: Vec<u32> = Vec::new();

        fn add_leafs(nodes: &mut Vec<u32>, rng: &mut SmallRng) {
            for _ in 0..8 {
                // Add Leaf
                nodes.push(u32::from_be_bytes([
                    // 0 - Node
                    // 1 - Empty Leaf
                    // 2 - Solid Leaf
                    if rng.gen_range(0..5) != 0 { 1 } else { 2 },
                    // Colour
                    rng.gen_range(120..250),
                    rng.gen_range(120..250),
                    rng.gen_range(120..250),
                ]));
            }
        }

        fn subdivie_leaf(nodes: &mut Vec<u32>, rng: &mut SmallRng, node: usize) -> usize {
            let i = nodes.len();
            add_leafs(nodes, rng);
            nodes[node] = i as u32;
            i
        }

        for _ in 0..4 {
            nodes.push(0);
        }

        for i in 0..NUMBER_OF_SUBDIVISIONS {
            if rng.gen_range(0..5) != 0 {
                subdivie_leaf(&mut nodes, &mut rng, i);
            }
        }

        data_len = nodes.len() as u32;

        CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            false,
            Vec::into_iter(nodes),
        )
        .unwrap()
    };

    let golden_ratio = 1.61803398875;
    let pi = 3.14159265359;

    let mut fibonacci_spiral = [[0.0; 4]; FIBONACHI_LENGTH];
    for i in 1..FIBONACHI_LENGTH + 1 {
        let pol = [
            (i as f32 / golden_ratio) % 1.0,
            i as f32 / FIBONACHI_LENGTH as f32,
        ];

        let theta = 2.0 * pi * pol[0];
        let radius = pol[1]; // Add .sqrt() for an even distribution

        let pos = [radius * theta.cos(), radius * theta.sin()];

        fibonacci_spiral[i - 1][0] = pos[0];
        fibonacci_spiral[i - 1][1] = pos[1];
    }
    // #endregion

    // #region Pipeline setup
    let render_pass = Arc::new(
        vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: swapchain.format(),
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        )
        .unwrap(),
    );

    let _ = include_str!("shader.vert");
    let _ = include_str!("graphics.frag");
    let _ = include_str!("post.frag");

    let vs = vs::Shader::load(device.clone()).unwrap();
    let graphics_fs = graphics_fs::Shader::load(device.clone()).unwrap();
    let post_fs = post_fs::Shader::load(device.clone()).unwrap();

    let graphics_pipeline = Arc::new(
        GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vs.main_entry_point(), ())
            .triangle_strip()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(graphics_fs.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap(),
    );

    let post_pipeline = Arc::new(
        GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vs.main_entry_point(), ())
            .triangle_strip()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(post_fs.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap(),
    );

    let mut dynamic_state = DynamicState {
        line_width: None,
        viewports: None,
        scissors: None,
        compare_mask: None,
        write_mask: None,
        reference: None,
    };

    let (mut framebuffers, mut graphics_image) = size_dependent_setup(
        &output_images,
        render_pass.clone(),
        &mut dynamic_state,
        device.clone(),
        queue.clone(),
    );
    // #endregion

    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    let time = SystemTime::now();
    let mut last_time = time.elapsed().unwrap();
    let mut cam_pos = [1.0, 1.0, 1.0];
    let mut cam_rot = [-2.19911, 0.785398];

    let light_pos = [1.2, 1.5, -1.9];

    let mut normal_bias = 0.000001;
    let mut speed = 0.02;
    let mut debug_setting = false;

    event_loop.run(move |event, _, control_flow| match event {
        winit::event::Event::DeviceEvent { event, .. } => match event {
            DeviceEvent::MouseMotion { delta } => {
                let sensitivity = 0.005;
                cam_rot[0] += -delta.1 as f32 * sensitivity;
                cam_rot[1] += delta.0 as f32 * sensitivity;
            }
            _ => {}
        },
        _ => {
            if input.update(&event) {
                // #region Update window
                if input.window_resized().is_some() {
                    recreate_swapchain = true;
                }

                previous_frame_end.as_mut().unwrap().cleanup_finished();

                if recreate_swapchain {
                    let dimensions: [u32; 2] = surface.window().inner_size().into();
                    let (new_swapchain, new_images) =
                        match swapchain.recreate_with_dimensions(dimensions) {
                            Ok(r) => r,
                            Err(SwapchainCreationError::UnsupportedDimensions) => return,
                            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                        };

                    swapchain = new_swapchain;

                    let (new_framebuffers, new_graphics_image) = size_dependent_setup(
                        &new_images,
                        render_pass.clone(),
                        &mut dynamic_state,
                        device.clone(),
                        queue.clone(),
                    );

                    framebuffers = new_framebuffers;
                    graphics_image = new_graphics_image;

                    recreate_swapchain = false;
                }
                // #endregion

                // #region Update Buffers
                let fps = (1.0
                    / ((time.elapsed().unwrap() - last_time).as_millis() as f64 / 1000.0))
                    as i32;
                println!("{:?}", fps);
                last_time = time.elapsed().unwrap();

                if input.key_pressed(VirtualKeyCode::Escape) {
                    surface.window().set_cursor_grab(false).unwrap();
                    surface.window().set_cursor_visible(true);
                }

                if input.mouse_pressed(0) {
                    surface.window().set_cursor_grab(true).unwrap();
                    surface.window().set_cursor_visible(false);
                }

                let dimensions = surface.window().inner_size().into();
                let time = time.elapsed().unwrap().as_millis() as f32;

                let forward = input.key_held(VirtualKeyCode::W) as i32
                    - input.key_held(VirtualKeyCode::S) as i32;
                let right = input.key_held(VirtualKeyCode::D) as i32
                    - input.key_held(VirtualKeyCode::A) as i32;
                let up = input.key_held(VirtualKeyCode::Space) as i32 - input.held_shift() as i32;

                let mut forward_vec = [0.0, 1.0, 0.0];
                forward_vec = rotate_x(forward_vec, cam_rot[0]);
                forward_vec = rotate_y(forward_vec, cam_rot[1]);

                let mut right_vec = [-1.0, 0.0, 0.0];
                right_vec = rotate_y(right_vec, cam_rot[1]);

                let up_vec = vec3_cross(forward_vec, right_vec);

                speed -= input.scroll_diff() / 100.0;
                if speed <= 0.0 {
                    speed = 0.0;
                }

                cam_pos = vec3_add(cam_pos, vec3_scale(forward_vec, forward as f32 * speed));
                cam_pos = vec3_add(cam_pos, vec3_scale(right_vec, right as f32 * speed));
                cam_pos = vec3_add(cam_pos, vec3_scale(up_vec, up as f32 * speed));

                if input.key_pressed(VirtualKeyCode::Up) {
                    normal_bias += 0.000001;
                }

                if input.key_pressed(VirtualKeyCode::Down) {
                    normal_bias -= 0.000001;
                }

                if input.key_pressed(VirtualKeyCode::P) {
                    debug_setting = !debug_setting;
                }

                if input.key_pressed(VirtualKeyCode::I) {
                    println!(
                        "pos: ({}, {}, {}), rot: ({}, {})",
                        cam_pos[0], cam_pos[1], cam_pos[2], cam_rot[0], cam_rot[1]
                    );
                }

                if input.key_released(VirtualKeyCode::Q) || input.quit() {
                    *control_flow = ControlFlow::Exit;
                    return;
                }
                // #endregion

                // #region Create Buffers
                let graphics_cam = graphics_fs::ty::Camera {
                    pos: cam_pos,
                    rot: cam_rot,
                    fov: 2.0,
                    max_depth: 10.0,
                    _dummy0: [0, 0, 0, 0],
                };

                let post_cam = post_fs::ty::Camera {
                    pos: cam_pos,
                    rot: cam_rot,
                    fov: 2.0,
                    max_depth: 10.0,
                    _dummy0: [0, 0, 0, 0],
                };

                let graphics_uniforms = graphics_fs::ty::Uniforms {
                    resolution: dimensions,
                    render_buffer_count: RENDER_BUFFER_COUNT,
                    time: time,
                    cam: graphics_cam,
                    light_pos: light_pos,
                    normal_bias: normal_bias,
                    data_len: data_len,
                };
                let graphics_set = Arc::new(
                    PersistentDescriptorSet::start(
                        graphics_pipeline.descriptor_set_layout(0).unwrap().clone(),
                    )
                    .add_buffer(graphics_uniform_buffer.next(graphics_uniforms).unwrap())
                    .unwrap()
                    .add_image(graphics_image.clone())
                    .unwrap()
                    .add_buffer(data_buffer.clone())
                    .unwrap()
                    .build()
                    .unwrap(),
                );

                let post_uniforms = post_fs::ty::Uniforms {
                    resolution: dimensions,
                    render_buffer_count: RENDER_BUFFER_COUNT,
                    time: time,
                    cam: post_cam,
                    light_pos: light_pos,
                    fibonacci_spiral: fibonacci_spiral,
                    debug_setting: debug_setting as u32,
                    _dummy0: [0, 0, 0, 0],
                };
                let post_set = Arc::new(
                    PersistentDescriptorSet::start(
                        post_pipeline.descriptor_set_layout(0).unwrap().clone(),
                    )
                    .add_buffer(post_uniform_buffer.next(post_uniforms).unwrap())
                    .unwrap()
                    .add_image(graphics_image.clone())
                    .unwrap()
                    .build()
                    .unwrap(),
                );
                // #endregion

                let (image_num, suboptimal, acquire_future) =
                    match swapchain::acquire_next_image(swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("Failed to acquire next image: {:?}", e),
                    };

                if suboptimal {
                    recreate_swapchain = true;
                }

                // #region Render Frame
                let clear_values = vec![[0.1255, 0.7373, 0.8471, 1.0].into()];

                let mut graphics_builder = AutoCommandBufferBuilder::primary_one_time_submit(
                    device.clone(),
                    queue.family(),
                )
                .unwrap();

                graphics_builder
                    .begin_render_pass(
                        framebuffers[image_num].clone(),
                        SubpassContents::Inline,
                        clear_values.clone(),
                    )
                    .unwrap()
                    .draw(
                        graphics_pipeline.clone(),
                        &dynamic_state,
                        vertex_buffer.clone(),
                        graphics_set,
                        (),
                        None,
                    )
                    .unwrap()
                    .draw(
                        post_pipeline.clone(),
                        &dynamic_state,
                        vertex_buffer.clone(),
                        post_set,
                        (),
                        None,
                    )
                    .unwrap()
                    .end_render_pass()
                    .unwrap();
                let graphics_command_buffer = graphics_builder.build().unwrap();

                let graphics_future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), graphics_command_buffer)
                    .unwrap()
                    .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                    .then_signal_fence_and_flush();
                // #endregion

                match graphics_future {
                    Ok(future) => {
                        previous_frame_end = Some(future.boxed());
                    }
                    Err(FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("Failed to flush future: {:?}", e);
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                }
            }
        }
    });
}

fn size_dependent_setup(
    output_images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState,
    device: Arc<Device>,
    queue: Arc<Queue>,
) -> (
    Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,
    Arc<StorageImage<Format>>,
) {
    let dimensions = output_images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0..1.0,
    };
    dynamic_state.viewports = Some(vec![viewport]);

    let output_framebuffers = output_images
        .iter()
        .map(|image| {
            Arc::new(
                Framebuffer::start(render_pass.clone())
                    .add(image.clone())
                    .unwrap()
                    .build()
                    .unwrap(),
            ) as Arc<dyn FramebufferAbstract + Send + Sync>
        })
        .collect::<Vec<_>>();

    let graphics_image = StorageImage::new(
        device.clone(),
        Dimensions::Dim2dArray {
            width: dimensions[0],
            height: dimensions[1],
            array_layers: RENDER_BUFFER_COUNT,
        },
        Format::R16G16B16A16Sfloat,
        Some(queue.family()),
    )
    .unwrap();

    (output_framebuffers, graphics_image)
}

fn rotate_x(vec: [f32; 3], angle: f32) -> [f32; 3] {
    [
        vec[0],
        vec[1] * angle.cos() - vec[2] * angle.sin(),
        vec[1] * angle.sin() + vec[2] * angle.cos(),
    ]
}

fn rotate_y(vec: [f32; 3], angle: f32) -> [f32; 3] {
    [
        vec[0] * angle.cos() + vec[2] * angle.sin(),
        vec[1],
        -vec[0] * angle.sin() + vec[2] * angle.cos(),
    ]
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shader.vert"
    }
}

mod graphics_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        include: [ "src" ],
        path: "src/graphics.frag"
    }
}

mod post_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        include: [ "src" ],
        path: "src/post.frag"
    }
}
