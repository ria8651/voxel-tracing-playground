use vulkano::buffer::cpu_pool::CpuBufferPool;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState, SubpassContents};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor::PipelineLayoutAbstract;
use vulkano::device::{Device, DeviceExtensions};
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass};
use vulkano::image::{ImageUsage, SwapchainImage};
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
use winit::event::{Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};
use winit_input_helper::WinitInputHelper;

use rand::Rng;
use std::sync::Arc;
use std::time::SystemTime;

fn main() {
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

    let (mut swapchain, images) = {
        let capabilities = surface.capabilities(physical).unwrap();
        let alpha = capabilities
            .supported_composite_alpha
            .iter()
            .next()
            .unwrap();
        let format = capabilities.supported_formats[0].0;
        let dimensions: [u32; 2] = surface.window().inner_size().into();
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

    // Buffer Setup

    let vertex_buffer = {
        #[derive(Default, Debug, Clone)]
        struct Vertex {
            position: [f32; 2],
        }
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

    let uniform_buffer = CpuBufferPool::<fs::ty::Uniforms>::new(device.clone(), BufferUsage::all());
    let data_len;
    let data_buffer = {
        // let root_node_bytes = [26u8, 201, 137, 0];
        // let root_node = u32::from_be_bytes(root_node_bytes);
        let mut rng = rand::thread_rng();
        let mut nodes: Vec<u32> = Vec::new();

        // for _ in 0..1 {
        //     if rng.gen::<bool>() {
        //         nodes.push(rng.gen_range(0..16777216));
        //     } else {
        //         nodes.push(u32::from_be_bytes([
        //             rng.gen_range(1..3),
        //             rng.gen_range(0..255),
        //             rng.gen_range(0..255),
        //             rng.gen_range(0..255),
        //         ]));
        //     }
        // }

        fn add_leafs(nodes: &mut Vec<u32>, rng: &mut rand::prelude::ThreadRng) {
            for _ in 0..8 {
                // Add Leaf
                nodes.push(u32::from_be_bytes([
                    rng.gen_range(1..3),
                    rng.gen_range(0..255),
                    rng.gen_range(0..255),
                    rng.gen_range(0..255),
                ]));
            }
        }

        fn subdivie_leaf(
            nodes: &mut Vec<u32>,
            rng: &mut rand::prelude::ThreadRng,
            node: usize,
        ) -> usize {
            let i = nodes.len();
            add_leafs(nodes, rng);
            nodes[node] = i as u32;
            i
        }

        add_leafs(&mut nodes, &mut rng);

        for i in 0..37448 {
            if rng.gen_range(0..15) != 0 {
                subdivie_leaf(&mut nodes, &mut rng, i);
            }
        }

        // for _ in 0..100 {
        //     loop {
        //         let rand = rng.gen_range(0..nodes.len());
        //         if nodes[rand] / 16777216 != 0 {
        //             subdivie_leaf(&mut nodes, &mut rng, rand);
        //             break;
        //         }
        //     }
        // }

        data_len = nodes.len();

        CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            false,
            Vec::into_iter(nodes),
        )
        .unwrap()
    };

    let _ = include_str!("shader.vert");
    let _ = include_str!("shader.frag");

    mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            path: "src/shader.vert"
        }
    }

    mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "src/shader.frag"
        }
    }
    let vs = vs::Shader::load(device.clone()).unwrap();
    let fs = fs::Shader::load(device.clone()).unwrap();

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

    let pipeline = Arc::new(
        GraphicsPipeline::start()
            .vertex_input_single_buffer()
            .vertex_shader(vs.main_entry_point(), ())
            .triangle_strip()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs.main_entry_point(), ())
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

    let mut framebuffers =
        window_size_dependent_setup(&images, render_pass.clone(), &mut dynamic_state);

    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    let time = SystemTime::now();
    let cam_pos = [4.0f32, 4.0, 4.0];
    let cam_rot = [0.9425, 0.785398];

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(_),
            ..
        } => {
            recreate_swapchain = true;
        }
        Event::RedrawEventsCleared => {
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
                framebuffers = window_size_dependent_setup(
                    &new_images,
                    render_pass.clone(),
                    &mut dynamic_state,
                );
                recreate_swapchain = false;
            }

            // Update Buffers

            let dimensions = surface.window().inner_size();
            let resolution = [dimensions.width as f32, dimensions.height as f32];
            let time = time.elapsed().unwrap().as_millis() as f32;

            if input.key_pressed(VirtualKeyCode::A) {
                println!("The 'A' key was pressed on the keyboard");
            }

            if input.key_released(VirtualKeyCode::Q) || input.quit() {
                *control_flow = ControlFlow::Exit;
                return;
            }

            let uniform_subbuffer = {
                let uniform_data = fs::ty::Uniforms {
                    resolution: resolution,
                    time: time,
                    cam_pos: cam_pos,
                    cam_rot: cam_rot,
                    data_len: data_len as u32,
                    _dummy0: [0, 0, 0, 0],
                    _dummy1: [0, 0, 0, 0],
                };

                uniform_buffer.next(uniform_data).unwrap()
            };

            let uniform_layout = pipeline.descriptor_set_layout(0).unwrap();
            let uniform_set = Arc::new(
                PersistentDescriptorSet::start(uniform_layout.clone())
                    .add_buffer(uniform_subbuffer)
                    .unwrap()
                    .build()
                    .unwrap(),
            );
            let data_layout = pipeline.descriptor_set_layout(1).unwrap();
            let data_set = Arc::new(
                PersistentDescriptorSet::start(data_layout.clone())
                    .add_buffer(data_buffer.clone())
                    .unwrap()
                    .build()
                    .unwrap(),
            );

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

            let clear_values = vec![[0.3333, 0.6745, 0.1059, 1.0].into()];
            let mut builder =
                AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family())
                    .unwrap();

            builder
                .begin_render_pass(
                    framebuffers[image_num].clone(),
                    SubpassContents::Inline,
                    clear_values,
                )
                .unwrap()
                .draw(
                    pipeline.clone(),
                    &dynamic_state,
                    vertex_buffer.clone(),
                    (uniform_set.clone(), data_set.clone()),
                    (),
                    None,
                )
                .unwrap()
                .end_render_pass()
                .unwrap();

            let command_buffer = builder.build().unwrap();

            let future = previous_frame_end
                .take()
                .unwrap()
                .join(acquire_future)
                .then_execute(queue.clone(), command_buffer)
                .unwrap()
                .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                .then_signal_fence_and_flush();

            match future {
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
        _ => (),
    });
}

fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState,
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0..1.0,
    };
    dynamic_state.viewports = Some(vec![viewport]);

    images
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
        .collect::<Vec<_>>()
}
