use std::time::Instant;

pub struct TimerEvent {
    name: String,
    duration: f64,
}

pub struct Timer {
    pub events: Vec<TimerEvent>,
    stamp: Instant,
    name: String,
    loop_data: Option<Vec<f64>>,
}

impl Timer {
    /// Create a new Timer instance.
    ///
    /// Parameters
    /// ----------
    /// name : &str
    ///     The name of this timer, used in the final report.
    ///
    /// Returns
    /// -------
    /// Timer
    ///     A new Timer instance with empty events and current timestamp.
    pub fn new(name: &str) -> Self {
        Timer {
            events: Vec::new(),
            stamp: Instant::now(),
            name: name.to_string(),
            loop_data: None,
        }
    }

    /// Record a timing event from the last timestamp.
    ///
    /// Records the elapsed time since the last call to `record()`, `loop_iteration()`,
    /// `loop_end()`, or timer creation, and resets the internal timestamp.
    ///
    /// Parameters
    /// ----------
    /// name : &str
    ///     The name of this timing event.
    pub fn record(&mut self, name: &str) {
        self.events.push(TimerEvent {
            name: name.to_string(),
            duration: self.stamp.elapsed().as_secs_f64(),
        });
        self.stamp = Instant::now();
    }

    /// Start timing a loop with multiple iterations.
    ///
    /// Initializes loop timing mode. After calling this, use `loop_iteration()`
    /// to record each iteration and `loop_end()` to generate statistics.
    #[allow(dead_code)]
    pub fn loop_start(&mut self) {
        assert!(
            self.loop_data.is_none(),
            "`Timer.loop_start` called while already in loop timing mode!"
        );
        self.loop_data = Some(Vec::new());
    }
    /// Record the completion of one loop iteration.
    ///
    /// Records the time elapsed since the last `loop_start()` or `loop_iteration()`
    /// call and resets the timestamp for the next iteration.
    ///
    /// Panics
    /// ------
    /// Panics if called before `loop_start()`.
    #[allow(dead_code)]
    pub fn loop_iteration(&mut self) {
        let iterations = self
            .loop_data
            .as_mut()
            .expect("`Timer.loop_iteration` called without `Timer.loop_start`!");
        let iteration_time = self.stamp.elapsed().as_secs_f64();
        iterations.push(iteration_time);
        self.stamp = Instant::now();
    }
    /// End loop timing and generate statistics.
    ///
    /// Records the final iteration time and creates a comprehensive timing event
    /// with statistics including total time, mean, standard deviation, iteration
    /// count, and slowest/fastest iteration times.
    ///
    /// Parameters
    /// ----------
    /// name : &str
    ///     The name for this loop timing event.
    ///
    /// Notes
    /// -----
    /// The generated message format is:
    /// "{name}: {total:.6}s total, {mean:.6} +/-{std:.6}s (mean/std) over {count} iterations (slowest: {slowest:.6}s fastest: {fastest:.6}s)"
    #[allow(dead_code)]
    pub fn loop_end(&mut self, name: &str) {
        if let Some(ref mut iterations) = self.loop_data {
            let iteration_time = self.stamp.elapsed().as_secs_f64();
            iterations.push(iteration_time);

            let count = iterations.len();
            let (total, slowest, fastest) = iterations.iter().fold(
                (0.0, f64::NEG_INFINITY, f64::INFINITY),
                |(sum, max_val, min_val), &x| (sum + x, max_val.max(x), min_val.min(x)),
            );
            let mean = total / count as f64;

            let variance: f64 =
                iterations.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / count as f64;
            let std = variance.sqrt();

            let message = format!(
                "{name}: {mean:.3} +/-{std:.3}s {count} iterations {slowest:.2}s-{fastest:.2}s"
            );

            self.events.push(TimerEvent {
                name: message,
                duration: total,
            });
        }
        self.loop_data = None;
        self.stamp = Instant::now();
    }

    /// Print a formatted report of all recorded timing events if
    /// we are in `cfg(test)` mode OR if the `RMESH_DEBUG` environment variable is set.
    pub fn print_conditionally(&self) {
        #[cfg(test)]
        {
            println!("{self}");
        }
        #[cfg(not(test))]
        {
            if std::env::var("RMESH_DEBUG").is_ok() {
                println!("{self}");
            }
        }
    }
}

/// Implements a formatted display for the Timer struct, showing each event's name,
/// duration, and percentage of total time.
impl std::fmt::Display for Timer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let total: f64 = self.events.iter().map(|e| e.duration).sum();
        let percent: Vec<f64> = self
            .events
            .iter()
            .map(|e| (e.duration / total) * 100.0)
            .collect();

        let longest_name: usize = self
            .events
            .iter()
            .map(|e| e.name.lines().map(|line| line.len()).max().unwrap_or(0))
            .max()
            .unwrap_or(0);

        writeln!(f, "Timer Report: {}", self.name)?;
        writeln!(f, "{total:.6}s Total")?;
        writeln!(f, "----------------")?;
        for (event, percent) in self.events.iter().zip(percent) {
            let lines: Vec<&str> = event.name.lines().collect();
            if lines.len() == 1 {
                writeln!(
                    f,
                    "{:width$} : {:.6}s ({:.2}%)",
                    event.name,
                    event.duration,
                    percent,
                    width = longest_name
                )?;
            } else {
                for line in &lines[..lines.len() - 1] {
                    writeln!(f, "{line}")?;
                }
                writeln!(
                    f,
                    "{:width$} : {:.6}s ({:.2}%)",
                    lines.last().unwrap(),
                    event.duration,
                    percent,
                    width = longest_name
                )?;
            }
        }
        writeln!(f, "----------------")?;
        Ok(())
    }
}

#[cfg(test)]
pub(crate) struct Profiler {
    guard: Option<pprof::ProfilerGuard<'static>>,
    report: Option<pprof::Report>,
}

#[cfg(test)]
impl Profiler {
    pub(crate) fn start() -> Self {
        let guard = pprof::ProfilerGuardBuilder::default()
            .frequency(4000)
            .blocklist(&["libc", "libgcc", "pthread", "vdso"])
            .build()
            .unwrap();
        Self {
            guard: Some(guard),
            report: None,
        }
    }

    pub(crate) fn stop(&mut self) {
        if let Some(guard) = self.guard.take() {
            self.report = guard.report().build().ok();
        }
    }
}

#[cfg(test)]
impl Profiler {
    /// Write a flamegraph SVG to the given path.
    pub(crate) fn flamegraph(&self, path: &std::path::Path) {
        if let Some(report) = &self.report {
            let file = std::fs::File::create(path).unwrap();
            report.flamegraph(file).unwrap();
            println!("flamegraph: {}", path.display());
        }
    }
}

#[cfg(test)]
impl std::fmt::Display for Profiler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let report = match &self.report {
            Some(r) => r,
            None => return write!(f, "Profiler: call .stop() first"),
        };

        let total_samples: isize = report.data.values().sum();
        if total_samples == 0 {
            return write!(f, "Profiler: 0 samples collected");
        }

        // Aggregate cumulative samples by file:line across all stack frames.
        // Each stack sample contributes to every frame in it (cumulative).
        let mut by_location: std::collections::HashMap<String, isize> =
            std::collections::HashMap::new();
        for (frames, count) in &report.data {
            let mut seen = std::collections::HashSet::new();
            for frame in &frames.frames {
                for sym in frame {
                    let name = sym.name();
                    // Skip non-project frames by function name
                    if !name.contains("rmesh") && !name.contains("i_overlay") {
                        continue;
                    }
                    let filename = sym.filename();
                    let lineno = sym.lineno();
                    let key = if lineno > 0 && filename != "Unknown" {
                        let short = filename
                            .rfind("crates/")
                            .or_else(|| filename.rfind("src/"))
                            .map(|i| &filename[i..])
                            .unwrap_or(&filename);
                        format!("{short}:{lineno} {name}")
                    } else {
                        name.to_string()
                    };
                    if seen.insert(key.clone()) {
                        *by_location.entry(key).or_default() += count;
                    }
                }
            }
        }

        let mut sorted: Vec<_> = by_location.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));

        writeln!(f, "Profiler: {total_samples} samples (cumulative)")?;
        writeln!(f, "{:-<80}", "")?;
        writeln!(f, "{:>5}  {:>5}  location", "%", "hits")?;
        writeln!(f, "{:-<80}", "")?;
        for (location, count) in sorted.iter().take(25) {
            let pct = *count as f64 / total_samples as f64 * 100.0;
            writeln!(f, "{pct:5.1}%  {count:>5}  {location}")?;
        }
        writeln!(f, "{:-<80}", "")?;
        Ok(())
    }
}

#[cfg(test)]
impl std::fmt::Debug for Profiler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_timer_events_and_display() {
        let mut timer = Timer::new("test_timer");

        thread::sleep(Duration::from_millis(10));
        timer.record("event_1");

        thread::sleep(Duration::from_millis(20));
        timer.record("event_2");

        timer.loop_start();
        let sleep_ms = [5, 15, 8, 12, 20, 10];
        for &ms in &sleep_ms {
            thread::sleep(Duration::from_millis(ms));
            timer.loop_iteration();
        }
        timer.loop_end("test_loop");

        thread::sleep(Duration::from_millis(15));
        timer.record("final_event");

        assert_eq!(timer.events.len(), 4);
        assert_eq!(timer.events[0].name, "event_1");
        assert_eq!(timer.events[1].name, "event_2");
        assert!(timer.events[2].name.starts_with("test_loop:"));
        assert!(timer.events[2].name.contains("7 iter"));
        assert_eq!(timer.events[3].name, "final_event");

        let display_output = format!("{timer}");
        assert!(display_output.contains("Timer Report: test_timer"));
        assert!(display_output.contains("Total"));
        assert!(display_output.contains("event_1"));
        assert!(display_output.contains("event_2"));
        assert!(display_output.contains("test_loop:"));
        assert!(display_output.contains("final_event"));
        assert!(display_output.contains("----------------"));
        assert!(display_output.contains("%)"));
        assert!(display_output.contains("+/-"));
        assert!(display_output.contains("iteration"));

        timer.print_conditionally();
    }
}
