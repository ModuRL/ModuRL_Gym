use minifb::{Window, WindowOptions};

pub struct Renderer {
    window: Window,
    buffer: Vec<u32>, // ARGB8888
    w: usize,
    h: usize,
}

impl Renderer {
    pub fn new(w: usize, h: usize, title: &str) -> Self {
        let window = Window::new(title, w, h, WindowOptions::default()).expect("create window");

        Self {
            window,
            buffer: vec![0; w * h],
            w,
            h,
        }
    }

    pub fn clear(&mut self, color: u32) {
        self.buffer.fill(color);
    }

    pub fn rect(&mut self, x: usize, y: usize, rw: usize, rh: usize, color: u32) {
        let x_end = (x + rw).min(self.w);
        let y_end = (y + rh).min(self.h);

        for row in y..y_end {
            let start = row * self.w + x;
            let end = row * self.w + x_end;
            self.buffer[start..end].fill(color);
        }
    }

    pub fn present(&mut self) {
        self.window
            .update_with_buffer(&self.buffer, self.w, self.h)
            .unwrap();
    }

    pub fn is_open(&self) -> bool {
        self.window.is_open()
    }

    pub fn quad(
        &mut self,
        p1: (f32, f32),
        p2: (f32, f32),
        p3: (f32, f32),
        p4: (f32, f32),
        color: u32,
    ) {
        let min_x = p1.0.min(p2.0.min(p3.0.min(p4.0))).max(0.0) as usize;
        let max_x = p1.0.max(p2.0.max(p3.0.max(p4.0))).min(self.w as f32 - 1.0) as usize;
        let min_y = p1.1.min(p2.1.min(p3.1.min(p4.1))).max(0.0) as usize;
        let max_y = p1.1.max(p2.1.max(p3.1.max(p4.1))).min(self.h as f32 - 1.0) as usize;

        if min_x >= max_x || min_y >= max_y {
            return;
        }

        // Only iterate over the bounding box, not the entire screen
        for row in min_y..=max_y {
            for col in min_x..=max_x {
                let px = col as f32 + 0.5;
                let py = row as f32 + 0.5;

                let b1 = (px - p2.0) * (p1.1 - p2.1) - (p1.0 - p2.0) * (py - p2.1) < 0.0;
                let b2 = (px - p3.0) * (p2.1 - p3.1) - (p2.0 - p3.0) * (py - p3.1) < 0.0;
                let b3 = (px - p4.0) * (p3.1 - p4.1) - (p3.0 - p4.0) * (py - p4.1) < 0.0;
                let b4 = (px - p1.0) * (p4.1 - p1.1) - (p4.0 - p1.0) * (py - p1.1) < 0.0;

                if (b1 == b2) && (b2 == b3) && (b3 == b4) {
                    self.buffer[row * self.w + col] = color;
                }
            }
        }
    }
}
