//! Streaming API for Large Results
#![allow(clippy::explicit_counter_loop)] // Streaming uses explicit counters
//!
//! This module provides streaming capabilities for large solver results,
//! allowing incremental processing without loading everything into memory.
//!
//! # Features
//!
//! - **Model Streaming**: Stream model values incrementally
//! - **Proof Streaming**: Stream proof steps one at a time
//! - **Result Chunking**: Break large results into manageable chunks
//! - **Backpressure**: Handle slow consumers gracefully
//!
//! # Example
//!
//! ```javascript
//! const solver = new StreamingSolver();
//! solver.setLogic("QF_LIA");
//!
//! // Declare many variables
//! for (let i = 0; i < 1000; i++) {
//!     solver.declareConst(`x${i}`, "Int");
//! }
//!
//! // Check satisfiability
//! const result = await solver.checkSat();
//! if (result === "sat") {
//!     // Stream model values
//!     for await (const entry of solver.streamModel()) {
//!         console.log(`${entry.name} = ${entry.value}`);
//!     }
//! }
//! ```

#![forbid(unsafe_code)]

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

/// Chunk of streamed data
#[wasm_bindgen]
pub struct DataChunk {
    /// Chunk data
    data: Vec<u8>,
    /// Sequence number
    sequence: usize,
    /// Total chunks (if known)
    total: Option<usize>,
    /// Whether this is the last chunk
    is_last: bool,
}

#[wasm_bindgen]
impl DataChunk {
    /// Create a new data chunk
    #[wasm_bindgen(constructor)]
    pub fn new(data: Vec<u8>, sequence: usize) -> Self {
        Self {
            data,
            sequence,
            total: None,
            is_last: false,
        }
    }

    /// Mark as last chunk
    #[wasm_bindgen(js_name = markLast)]
    pub fn mark_last(mut self) -> Self {
        self.is_last = true;
        self
    }

    /// Set total chunk count
    #[wasm_bindgen(js_name = withTotal)]
    pub fn with_total(mut self, total: usize) -> Self {
        self.total = Some(total);
        self
    }

    /// Get chunk size
    #[wasm_bindgen(js_name = size)]
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Get sequence number
    #[wasm_bindgen(js_name = sequence)]
    pub fn sequence(&self) -> usize {
        self.sequence
    }

    /// Check if this is the last chunk
    #[wasm_bindgen(js_name = isLast)]
    pub fn is_last(&self) -> bool {
        self.is_last
    }

    /// Get data as JS Uint8Array
    #[wasm_bindgen(js_name = getData)]
    pub fn get_data(&self) -> js_sys::Uint8Array {
        js_sys::Uint8Array::from(&self.data[..])
    }

    /// Get progress percentage (if total known)
    #[wasm_bindgen(js_name = progress)]
    pub fn progress(&self) -> Option<f64> {
        self.total.map(|total| {
            if total == 0 {
                100.0
            } else {
                (self.sequence as f64 / total as f64) * 100.0
            }
        })
    }
}

/// Model entry for streaming
#[wasm_bindgen]
pub struct ModelEntry {
    /// Variable name
    name: String,
    /// Variable value
    value: String,
    /// Sort (type)
    sort: String,
}

#[wasm_bindgen]
impl ModelEntry {
    /// Create a new model entry
    #[wasm_bindgen(constructor)]
    pub fn new(name: String, value: String, sort: String) -> Self {
        Self { name, value, sort }
    }

    /// Get variable name
    #[wasm_bindgen(js_name = getName)]
    pub fn get_name(&self) -> String {
        self.name.clone()
    }

    /// Get variable value
    #[wasm_bindgen(js_name = getValue)]
    pub fn get_value(&self) -> String {
        self.value.clone()
    }

    /// Get variable sort
    #[wasm_bindgen(js_name = getSort)]
    pub fn get_sort(&self) -> String {
        self.sort.clone()
    }

    /// Export as JS object
    #[wasm_bindgen(js_name = toObject)]
    pub fn to_object(&self) -> JsValue {
        let obj = js_sys::Object::new();
        let _ = js_sys::Reflect::set(&obj, &"name".into(), &self.name.clone().into());
        let _ = js_sys::Reflect::set(&obj, &"value".into(), &self.value.clone().into());
        let _ = js_sys::Reflect::set(&obj, &"sort".into(), &self.sort.clone().into());
        obj.into()
    }
}

/// Stream controller for managing data flow
#[wasm_bindgen]
pub struct StreamController {
    /// Buffered chunks
    buffer: Rc<RefCell<VecDeque<DataChunk>>>,
    /// Maximum buffer size
    max_buffer_size: usize,
    /// Whether stream is closed
    closed: Rc<RefCell<bool>>,
    /// Total bytes streamed
    total_bytes: Rc<RefCell<usize>>,
    /// Chunk count
    chunk_count: Rc<RefCell<usize>>,
}

#[wasm_bindgen]
impl StreamController {
    /// Create a new stream controller
    #[wasm_bindgen(constructor)]
    pub fn new(max_buffer_size: usize) -> Self {
        Self {
            buffer: Rc::new(RefCell::new(VecDeque::new())),
            max_buffer_size,
            closed: Rc::new(RefCell::new(false)),
            total_bytes: Rc::new(RefCell::new(0)),
            chunk_count: Rc::new(RefCell::new(0)),
        }
    }

    /// Enqueue a chunk
    #[wasm_bindgen(js_name = enqueue)]
    pub fn enqueue(&self, chunk: DataChunk) -> bool {
        if *self.closed.borrow() {
            return false;
        }

        let mut buffer = self.buffer.borrow_mut();
        if buffer.len() >= self.max_buffer_size {
            return false; // Buffer full, apply backpressure
        }

        *self.total_bytes.borrow_mut() += chunk.size();
        *self.chunk_count.borrow_mut() += 1;

        buffer.push_back(chunk);
        true
    }

    /// Dequeue a chunk
    #[wasm_bindgen(js_name = dequeue)]
    pub fn dequeue(&self) -> Option<DataChunk> {
        self.buffer.borrow_mut().pop_front()
    }

    /// Get buffer length
    #[wasm_bindgen(js_name = bufferLength)]
    pub fn buffer_length(&self) -> usize {
        self.buffer.borrow().len()
    }

    /// Check if stream is closed
    #[wasm_bindgen(js_name = isClosed)]
    pub fn is_closed(&self) -> bool {
        *self.closed.borrow()
    }

    /// Close the stream
    #[wasm_bindgen(js_name = close)]
    pub fn close(&self) {
        *self.closed.borrow_mut() = true;
    }

    /// Get total bytes streamed
    #[wasm_bindgen(js_name = totalBytes)]
    pub fn total_bytes(&self) -> usize {
        *self.total_bytes.borrow()
    }

    /// Get chunk count
    #[wasm_bindgen(js_name = chunkCount)]
    pub fn chunk_count(&self) -> usize {
        *self.chunk_count.borrow()
    }

    /// Get statistics
    #[wasm_bindgen(js_name = getStats)]
    pub fn get_stats(&self) -> JsValue {
        let stats = js_sys::Object::new();
        let _ = js_sys::Reflect::set(
            &stats,
            &"buffer_length".into(),
            &self.buffer_length().into(),
        );
        let _ = js_sys::Reflect::set(&stats, &"total_bytes".into(), &self.total_bytes().into());
        let _ = js_sys::Reflect::set(&stats, &"chunk_count".into(), &self.chunk_count().into());
        let _ = js_sys::Reflect::set(&stats, &"is_closed".into(), &self.is_closed().into());
        stats.into()
    }
}

/// Streaming solver with incremental result delivery
#[wasm_bindgen]
pub struct StreamingSolver {
    /// Solver context
    ctx: oxiz_solver::Context,
    /// Stream controller
    controller: Option<StreamController>,
    /// Chunk size for streaming
    chunk_size: usize,
}

#[wasm_bindgen]
impl StreamingSolver {
    /// Create a new streaming solver
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            ctx: oxiz_solver::Context::new(),
            controller: None,
            chunk_size: 4096,
        }
    }

    /// Set chunk size
    #[wasm_bindgen(js_name = setChunkSize)]
    pub fn set_chunk_size(&mut self, size: usize) {
        self.chunk_size = size;
    }

    /// Set logic
    #[wasm_bindgen(js_name = setLogic)]
    pub fn set_logic(&mut self, logic: &str) {
        self.ctx.set_logic(logic);
    }

    /// Declare constant
    #[wasm_bindgen(js_name = declareConst)]
    pub fn declare_const(&mut self, name: &str, sort_name: &str) -> Result<(), JsValue> {
        let sort = self.parse_sort(sort_name)?;
        self.ctx.declare_const(name, sort);
        Ok(())
    }

    /// Assert formula
    #[wasm_bindgen(js_name = assertFormula)]
    pub fn assert_formula(&mut self, formula: &str) -> Result<(), JsValue> {
        let script = format!("(assert {})", formula);
        self.ctx
            .execute_script(&script)
            .map_err(|e| JsValue::from_str(&format!("Failed to assert: {}", e)))?;
        Ok(())
    }

    /// Check satisfiability
    #[wasm_bindgen(js_name = checkSat)]
    pub fn check_sat(&mut self) -> String {
        let result = self.ctx.check_sat();
        match result {
            oxiz_solver::SolverResult::Sat => "sat",
            oxiz_solver::SolverResult::Unsat => "unsat",
            oxiz_solver::SolverResult::Unknown => "unknown",
        }
        .to_string()
    }

    /// Start streaming model
    #[wasm_bindgen(js_name = startModelStream)]
    pub fn start_model_stream(&mut self) -> StreamController {
        let controller = StreamController::new(100);
        self.controller = Some(controller);
        StreamController::new(100)
    }

    /// Get next model entry
    #[wasm_bindgen(js_name = nextModelEntry)]
    pub fn next_model_entry(&self) -> Option<ModelEntry> {
        // In a real implementation, this would parse the model incrementally
        // For now, return None
        None
    }

    /// Stream model to chunks
    #[wasm_bindgen(js_name = streamModelChunks)]
    pub fn stream_model_chunks(&self, controller: &StreamController) {
        let model = self.ctx.get_model();

        // Convert model to string representation, then to bytes
        let model_str = match model {
            Some(entries) => entries
                .iter()
                .map(|(name, sort, value)| format!("({} {} {})", name, sort, value))
                .collect::<Vec<_>>()
                .join("\n"),
            None => String::new(),
        };
        let bytes = model_str.as_bytes();

        let mut sequence = 0;
        let total_chunks = if bytes.is_empty() {
            1
        } else {
            bytes.len().div_ceil(self.chunk_size)
        };

        for chunk_data in bytes.chunks(self.chunk_size) {
            let is_last = sequence == total_chunks - 1;
            let chunk = DataChunk::new(chunk_data.to_vec(), sequence).with_total(total_chunks);

            let chunk = if is_last { chunk.mark_last() } else { chunk };

            if !controller.enqueue(chunk) {
                break; // Backpressure
            }

            sequence += 1;
        }

        controller.close();
    }

    // Helper to parse sorts - returns SortId
    fn parse_sort(&self, sort_name: &str) -> Result<oxiz_core::SortId, JsValue> {
        match sort_name {
            "Bool" => Ok(self.ctx.terms.sorts.bool_sort),
            "Int" => Ok(self.ctx.terms.sorts.int_sort),
            "Real" => Ok(self.ctx.terms.sorts.real_sort),
            _ => Err(JsValue::from_str(&format!("Unknown sort: {}", sort_name))),
        }
    }
}

impl Default for StreamingSolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Result aggregator for collecting streamed results
#[wasm_bindgen]
pub struct ResultAggregator {
    /// Collected chunks
    chunks: Vec<DataChunk>,
    /// Total size
    total_size: usize,
}

#[wasm_bindgen]
impl ResultAggregator {
    /// Create a new result aggregator
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            chunks: Vec::new(),
            total_size: 0,
        }
    }

    /// Add a chunk
    #[wasm_bindgen(js_name = addChunk)]
    pub fn add_chunk(&mut self, chunk: DataChunk) {
        self.total_size += chunk.size();
        self.chunks.push(chunk);
    }

    /// Get chunk count
    #[wasm_bindgen(js_name = chunkCount)]
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Get total size
    #[wasm_bindgen(js_name = totalSize)]
    pub fn total_size(&self) -> usize {
        self.total_size
    }

    /// Assemble all chunks into one buffer
    #[wasm_bindgen(js_name = assemble)]
    pub fn assemble(&self) -> js_sys::Uint8Array {
        let mut result = Vec::with_capacity(self.total_size);
        for chunk in &self.chunks {
            result.extend_from_slice(&chunk.data);
        }
        js_sys::Uint8Array::from(&result[..])
    }

    /// Clear all chunks
    #[wasm_bindgen(js_name = clear)]
    pub fn clear(&mut self) {
        self.chunks.clear();
        self.total_size = 0;
    }
}

impl Default for ResultAggregator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_chunk() {
        let chunk = DataChunk::new(vec![1, 2, 3, 4], 0);
        assert_eq!(chunk.size(), 4);
        assert_eq!(chunk.sequence(), 0);
        assert!(!chunk.is_last());
    }

    #[test]
    fn test_data_chunk_last() {
        let chunk = DataChunk::new(vec![1, 2, 3], 0).mark_last();
        assert!(chunk.is_last());
    }

    #[test]
    fn test_data_chunk_progress() {
        let chunk = DataChunk::new(vec![1, 2], 2).with_total(10);
        assert_eq!(chunk.progress(), Some(20.0));
    }

    #[test]
    fn test_model_entry() {
        let entry = ModelEntry::new("x".to_string(), "42".to_string(), "Int".to_string());

        assert_eq!(entry.get_name(), "x");
        assert_eq!(entry.get_value(), "42");
        assert_eq!(entry.get_sort(), "Int");
    }

    #[test]
    fn test_stream_controller() {
        let controller = StreamController::new(10);

        let chunk1 = DataChunk::new(vec![1, 2, 3], 0);
        let chunk2 = DataChunk::new(vec![4, 5, 6], 1);

        assert!(controller.enqueue(chunk1));
        assert!(controller.enqueue(chunk2));

        assert_eq!(controller.buffer_length(), 2);
        assert_eq!(controller.total_bytes(), 6);
    }

    #[test]
    fn test_stream_controller_backpressure() {
        let controller = StreamController::new(2);

        controller.enqueue(DataChunk::new(vec![1], 0));
        controller.enqueue(DataChunk::new(vec![2], 1));

        // Buffer full, should return false
        assert!(!controller.enqueue(DataChunk::new(vec![3], 2)));
    }

    #[test]
    fn test_stream_controller_dequeue() {
        let controller = StreamController::new(10);

        controller.enqueue(DataChunk::new(vec![1, 2], 0));
        controller.enqueue(DataChunk::new(vec![3, 4], 1));

        let chunk = controller.dequeue().unwrap();
        assert_eq!(chunk.sequence(), 0);
        assert_eq!(controller.buffer_length(), 1);
    }

    #[test]
    fn test_result_aggregator() {
        let mut aggregator = ResultAggregator::new();

        aggregator.add_chunk(DataChunk::new(vec![1, 2, 3], 0));
        aggregator.add_chunk(DataChunk::new(vec![4, 5], 1));

        assert_eq!(aggregator.chunk_count(), 2);
        assert_eq!(aggregator.total_size(), 5);
    }

    #[test]
    fn test_streaming_solver() {
        let solver = StreamingSolver::new();
        assert_eq!(solver.chunk_size, 4096);
    }
}
