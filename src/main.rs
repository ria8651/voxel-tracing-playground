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

use na::base::Matrix;
use na::geometry::Rotation3;
use nalgebra as na;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::sync::Arc;
use std::time::SystemTime;

type Vector3 = na::Vector3<f32>;
type Matrix4 = na::Matrix4<f32>;

// static NUMBER_OF_SUBDIVISIONS: usize = 100000;
static RENDER_BUFFER_COUNT: u32 = 3;
const FIBONACHI_LENGTH: usize = 20;

fn main() {
    // #region Initilisation
    let instance = Instance::new(None, &vulkano_win::required_extensions(), None).unwrap();
    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();

    // println!("{}", physical.limits().max_image_dimension_2d());

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

    let pass1_uniform_buffer =
        CpuBufferPool::<pass1::ty::Uniforms>::new(device.clone(), BufferUsage::all());
    let pass2_uniform_buffer =
        CpuBufferPool::<pass2::ty::Uniforms>::new(device.clone(), BufferUsage::all());
    let pass3_uniform_buffer =
        CpuBufferPool::<pass3::ty::Uniforms>::new(device.clone(), BufferUsage::all());

    let mut rng: SmallRng = SeedableRng::seed_from_u64(0);
    let mut nodes: Vec<u64> = Vec::new();
    let mut voxels: Vec<u32> = Vec::new();

    fn create_node(child_mask: u8, child_pointer: u32) -> u64 {
        (child_pointer as u64) | ((child_mask as u64) << 32)
    }

    fn create_leaf(rng: &mut SmallRng) -> u32 {
        u32::from_be_bytes([
            // Colour
            rng.gen_range(120..250),
            rng.gen_range(120..250),
            rng.gen_range(120..250),
            0,
        ])
    }

    fn add_nodes(child_mask: u8, nodes: &mut Vec<u64>) {
        for i in 0..8 {
            let bit = (child_mask >> i) & 1;
            if bit != 0 {
                nodes.push(create_node(0, 0));
            }
        }
    }

    nodes.push(create_node(0, 0));
    for i in 0..8 {
        if i == 0 {
            let child_mask = 254;
            let child_pointer = nodes.len() as u32;
            nodes[i] = create_node(child_mask, child_pointer);

            add_nodes(child_mask, &mut nodes);
        } else {
            let child_mask = 0;
            let child_pointer = 2147483648 + voxels.len() as u32;
            nodes[i] = create_node(child_mask, child_pointer);
            
            voxels.push(create_leaf(&mut rng));
        }
    }

    for i in 0..voxels.len() {
        println!("r{}: {}", i, voxels[i]);
    }

    let node_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        false,
        Vec::into_iter(nodes),
    )
    .unwrap();

    let voxel_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        false,
        Vec::into_iter(voxels),
    )
    .unwrap();

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
    let _ = include_str!("pass1.frag");
    let _ = include_str!("pass2.frag");
    let _ = include_str!("pass3.frag");
    let _ = include_str!("common.glsl");

    let vs = vs::Shader::load(device.clone()).unwrap();
    let pass1 = pass1::Shader::load(device.clone()).unwrap();
    let pass2 = pass2::Shader::load(device.clone()).unwrap();
    let pass3 = pass3::Shader::load(device.clone()).unwrap();

    let pass1_pipeline = Arc::new(
        GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vs.main_entry_point(), ())
            .triangle_strip()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(pass1.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap(),
    );

    let pass2_pipeline = Arc::new(
        GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vs.main_entry_point(), ())
            .triangle_strip()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(pass2.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap(),
    );

    let pass3_pipeline = Arc::new(
        GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vs.main_entry_point(), ())
            .triangle_strip()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(pass3.main_entry_point(), ())
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

    let (mut framebuffers, mut pass_images) = size_dependent_setup(
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
    // let mut last_time = time.elapsed().unwrap();

    let mut cam_pos = Vector3::new(1.0, 1.0, 1.0);
    let mut cam_target = Vector3::new(-1.0, -1.0, -1.0).normalize();
    let mut cam_right = Vector3::new(0.0, 0.0, 1.0).normalize();

    let mut camera_matrix = Matrix4::default();
    let mut camera_matrix_last = Matrix4::default();

    let light_pos = Vector3::new(1.2, 1.5, -1.9);

    let mut normal_bias = 0.000001;
    let mut speed = 0.02;
    let mut debug_setting = false;

    event_loop.run(move |event, _, control_flow| match event {
        winit::event::Event::DeviceEvent { event, .. } => match event {
            DeviceEvent::MouseMotion { delta } => {
                let sensitivity = 0.005;
                let right = delta.0 as f32 * sensitivity;
                let up = delta.1 as f32 * sensitivity;

                let right_rotation_matirx = Rotation3::new(Vector3::new(0.0, 1.0, 0.0) * right);
                cam_right = right_rotation_matirx * cam_right;

                let up_rotation_matirx = Rotation3::new(cam_right * up);
                cam_target = up_rotation_matirx * right_rotation_matirx * cam_target;
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

                    let (new_framebuffers, new_pass_images) = size_dependent_setup(
                        &new_images,
                        render_pass.clone(),
                        &mut dynamic_state,
                        device.clone(),
                        queue.clone(),
                    );

                    framebuffers = new_framebuffers;
                    pass_images = new_pass_images;

                    recreate_swapchain = false;
                }
                // #endregion

                // #region Update Buffers
                // let fps = (1.0
                //     / ((time.elapsed().unwrap() - last_time).as_millis() as f64 / 1000.0))
                //     as i32;
                // last_time = time.elapsed().unwrap();

                // println!("{:?}", fps);

                if input.key_pressed(VirtualKeyCode::Escape) {
                    surface.window().set_cursor_grab(false).unwrap();
                    surface.window().set_cursor_visible(true);
                }

                if input.mouse_pressed(0) {
                    surface.window().set_cursor_grab(true).unwrap();
                    surface.window().set_cursor_visible(false);
                }

                let dimensions: [u32; 2] = surface.window().inner_size().into();
                let time = time.elapsed().unwrap().as_millis() as f32;

                let forward = input.key_held(VirtualKeyCode::W) as i32
                    - input.key_held(VirtualKeyCode::S) as i32;
                let right = input.key_held(VirtualKeyCode::D) as i32
                    - input.key_held(VirtualKeyCode::A) as i32;
                let up = input.key_held(VirtualKeyCode::Space) as i32 - input.held_shift() as i32;

                let forward_vec = cam_target;
                let right_vec = cam_right;
                let up_vec = Vector3::cross(&forward_vec, &right_vec);

                speed *= 2.0 / (1.0 + 2.0_f32.powf(0.2 * input.scroll_diff())); // Sigmoid

                cam_pos += forward_vec * forward as f32 * speed
                    + right_vec * right as f32 * speed
                    + up_vec * up as f32 * speed;

                if !debug_setting {
                    camera_matrix_last = camera_matrix;
                }

                let view_matrix = Matrix::face_towards(
                    &cam_pos.into(),
                    &(cam_pos + cam_target).into(),
                    &Vector3::new(0.0, 1.0, 0.0),
                );

                // let aspect = dimensions[0] as f32 / dimensions[1] as f32;
                // let projection_matrix = create_projection_matrix(2.0, aspect, 0.01, 1.0);

                camera_matrix = view_matrix;

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
                        "pos: ({:.1}, {:.1}, {:.1}), forward: ({:.1}, {:.1}, {:.1}), right: ({:.1}, {:.1}, {:.1})",
                        cam_pos.x,
                        cam_pos.y,
                        cam_pos.z,
                        cam_target.x,
                        cam_target.y,
                        cam_target.z,
                        cam_right.x,
                        cam_right.y,
                        cam_right.z
                    );
                }

                if input.key_released(VirtualKeyCode::Q) || input.quit() {
                    *control_flow = ControlFlow::Exit;
                    return;
                }
                // #endregion

                // #region Create Buffers
                // dont use me i'm slow and bad! (but good for debuging) - shader_debug
                // let debug_image = StorageImage::new(
                //     device.clone(),
                //     Dimensions::Dim2d {
                //         width: 50,
                //         height: 1,
                //     },
                //     Format::R32Sfloat,
                //     Some(queue.family()),
                // )
                // .unwrap();

                let camera_matrix_inverse = camera_matrix.try_inverse().unwrap_or(Matrix4::identity());
                let camera_matrix_last_inverse = camera_matrix_last.try_inverse().unwrap_or(Matrix4::identity());

                let pass1_cam = pass1::ty::Camera {
                    camera: camera_matrix.into(),
                    camera_last: camera_matrix_last.into(),
                    camera_inverse: camera_matrix_inverse.into(),
                    camera_last_inverse: camera_matrix_last_inverse.into(),
                    fov: 2.0,
                    max_depth: 10.0,
                };

                let pass2_cam = pass2::ty::Camera {
                    camera: camera_matrix.into(),
                    camera_last: camera_matrix_last.into(),
                    camera_inverse: camera_matrix_inverse.into(),
                    camera_last_inverse: camera_matrix_last_inverse.into(),
                    fov: 2.0,
                    max_depth: 10.0,
                };

                let pass3_cam = pass3::ty::Camera {
                    camera: camera_matrix.into(),
                    camera_last: camera_matrix_last.into(),
                    camera_inverse: camera_matrix_inverse.into(),
                    camera_last_inverse: camera_matrix_last_inverse.into(),
                    fov: 2.0,
                    max_depth: 10.0,
                };

                let pass1_uniforms = pass1::ty::Uniforms {
                    resolution: dimensions,
                    render_buffer_count: RENDER_BUFFER_COUNT,
                    cam: pass1_cam,
                    debug_setting: debug_setting as u32,
                    _dummy0: [0, 0, 0, 0],
                    _dummy1: [0, 0, 0, 0, 0, 0, 0, 0],
                };
                let pass1_set = Arc::new(
                    PersistentDescriptorSet::start(
                        pass1_pipeline.descriptor_set_layout(0).unwrap().clone(),
                    )
                    .add_buffer(pass1_uniform_buffer.next(pass1_uniforms).unwrap())
                    .unwrap()
                    .add_image(pass_images.clone())
                    .unwrap()
                    .build()
                    .unwrap(),
                );

                let pass2_uniforms = pass2::ty::Uniforms {
                    resolution: dimensions,
                    render_buffer_count: RENDER_BUFFER_COUNT,
                    time: time,
                    cam: pass2_cam.into(),
                    light_pos: light_pos.into(),
                    normal_bias: normal_bias,
                    debug_setting: debug_setting as u32,
                    _dummy0: [0, 0, 0, 0, 0, 0, 0, 0],
                };
                let pass2_set = Arc::new(
                    PersistentDescriptorSet::start(
                        pass2_pipeline.descriptor_set_layout(0).unwrap().clone(),
                    )
                    .add_buffer(pass2_uniform_buffer.next(pass2_uniforms).unwrap())
                    .unwrap()
                    .add_image(pass_images.clone())
                    .unwrap()
                    .add_buffer(node_buffer.clone())
                    .unwrap()
                    .add_buffer(voxel_buffer.clone())
                    .unwrap()
                    // For debuging ONLY - shader_debug
                    // .add_image(debug_image.clone())
                    // .unwrap()
                    .build()
                    .unwrap(),
                );

                let pass3_uniforms = pass3::ty::Uniforms {
                    resolution: dimensions,
                    render_buffer_count: RENDER_BUFFER_COUNT,
                    time: time,
                    cam: pass3_cam.into(),
                    light_pos: light_pos.into(),
                    fibonacci_spiral: fibonacci_spiral,
                    debug_setting: debug_setting as u32,
                    _dummy0: [0, 0, 0, 0, 0, 0, 0, 0],
                    _dummy1: [0, 0, 0, 0],
                };
                let pass3_set = Arc::new(
                    PersistentDescriptorSet::start(
                        pass3_pipeline.descriptor_set_layout(0).unwrap().clone(),
                    )
                    .add_buffer(pass3_uniform_buffer.next(pass3_uniforms).unwrap())
                    .unwrap()
                    .add_image(pass_images.clone())
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

                let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(
                    device.clone(),
                    queue.family(),
                )
                .unwrap();

                // slow and bad! only for debuging! - shader_debug
                // let debug_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, (0 .. 50).map(|_| 0f32)).expect("failed to create buffer");

                builder
                    .begin_render_pass(
                        framebuffers[image_num].clone(),
                        SubpassContents::Inline,
                        clear_values.clone(),
                    )
                    .unwrap()
                    .draw(
                        pass1_pipeline.clone(),
                        &dynamic_state,
                        vertex_buffer.clone(),
                        pass1_set,
                        (),
                        None,
                    )
                    .unwrap()
                    .draw(
                        pass2_pipeline.clone(),
                        &dynamic_state,
                        vertex_buffer.clone(),
                        pass2_set,
                        (),
                        None,
                    )
                    .unwrap()
                    .draw(
                        pass3_pipeline.clone(),
                        &dynamic_state,
                        vertex_buffer.clone(),
                        pass3_set,
                        (),
                        None,
                    )
                    .unwrap()
                    .end_render_pass()
                    .unwrap()
                    // ONLY FOR DEBUGING - shader_debug
                    // .copy_image_to_buffer(debug_image.clone(), debug_buffer.clone())
                    // .unwrap()
                    ;
                let command_buffer = builder.build().unwrap();
                
                let graphics_future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                    .then_signal_fence_and_flush()
                    // Kills program for debuging - shader_debug
                    // .unwrap()
                    // .wait(None)
                    // .unwrap()
                    ;
                
                // shader_debug
                // let buffer_content = debug_buffer.read().unwrap();
                // for i in 0..9 {
                //     println!("{}: {}", i, buffer_content[i]);
                // }
                
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
                // #endregion
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

    let pass_images = StorageImage::new(
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

    (output_framebuffers, pass_images)
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shader.vert"
    }
}

mod pass1 {
    vulkano_shaders::shader! {
        ty: "fragment",
        include: [ "src" ],
        path: "src/pass1.frag"
    }
}

mod pass2 {
    vulkano_shaders::shader! {
        ty: "fragment",
        include: [ "src" ],
        path: "src/pass2.frag"
    }
}

mod pass3 {
    vulkano_shaders::shader! {
        ty: "fragment",
        include: [ "src" ],
        path: "src/pass3.frag"
    }
}