//! Host services shared by every code generator: metered device allocation and
//! output publication, exported under the names in [`crate::runtime::module_abi`].

use crate::runtime::{GpuDialect, module_abi};

/// Render the services for `dialect`. They are host-only and must follow the
/// standard headers; `catena_runtime_allocate` and `catena_runtime_publish` are the
/// entry points generated host code calls.
pub(crate) fn render(dialect: GpuDialect) -> String {
    format!(
        r#"#ifndef {guard}
__host__ static inline void catena_runtime_reject(const char *reason) {{
    fprintf(stderr, "catena runtime rejected generated code: %s\n", reason);
    fflush(stderr);
    __builtin_trap();
}}

// Generated allocations are charged cumulatively within one entry-point
// invocation. Freeing a buffer does not refund budget: this deliberately caps
// allocation churn as well as peak live bytes.
static thread_local uint64_t catena_runtime_allocation_budget = UINT64_MAX;
static thread_local uint64_t catena_runtime_allocated_bytes = 0;

extern "C" __host__ void {budget_begin}(uint64_t byte_limit) {{
    catena_runtime_allocation_budget = byte_limit;
    catena_runtime_allocated_bytes = 0;
}}

extern "C" __host__ void {budget_end}() {{
    catena_runtime_allocation_budget = UINT64_MAX;
    catena_runtime_allocated_bytes = 0;
}}

// Allocate `element_count * element_size` device bytes; null when that is zero.
__host__ static inline void *catena_runtime_allocate(uint64_t element_count, uint64_t element_size) {{
    if (element_size != 0 && element_count > UINT64_MAX / element_size) {{
        catena_runtime_reject("allocation byte count overflowed");
    }}
    uint64_t byte_len = element_count * element_size;
    if (byte_len > SIZE_MAX) {{
        catena_runtime_reject("allocation byte count exceeds size_t");
    }}
    if (catena_runtime_allocated_bytes > catena_runtime_allocation_budget
        || byte_len > catena_runtime_allocation_budget - catena_runtime_allocated_bytes) {{
        catena_runtime_reject("allocation budget exceeded");
    }}
    void *data = nullptr;
    if (byte_len == 0) {{
        return data;
    }}
    {error_type} error = {device_alloc}(&data, (size_t)byte_len);
    if (error != {success}) {{
        fprintf(stderr, "catena GPU error: %s\n", {error_string}(error));
        fflush(stderr);
        __builtin_trap();
    }}
    catena_runtime_allocated_bytes += byte_len;
    return data;
}}

// Output publication is scoped to one invocation, like allocation accounting.
// The callback runs synchronously on this host thread while the owner is live.
typedef uint8_t (*catena_publication_callback_t)(void *, uint64_t, const void *, uint64_t);
static thread_local void *catena_runtime_publication_context = nullptr;
static thread_local catena_publication_callback_t catena_runtime_publication_callback = nullptr;

extern "C" __host__ void {set_publication}(void *context, catena_publication_callback_t callback) {{
    catena_runtime_publication_context = context;
    catena_runtime_publication_callback = callback;
}}

// Publish the first `bytes` of output `output`, which currently holds `len` bytes at `data`.
__host__ static inline void catena_runtime_publish(
    const void *data, uint64_t len, uint64_t output, uint64_t bytes
) {{
    if (bytes > len || catena_runtime_publication_callback == nullptr) {{
        catena_runtime_reject("invalid output publication");
    }}
    if (!catena_runtime_publication_callback(catena_runtime_publication_context, output, data, bytes)) {{
        catena_runtime_reject("output publication cancelled");
    }}
}}
#endif
"#,
        guard = dialect.device_compile_guard(),
        budget_begin = module_abi::ALLOCATION_BUDGET_BEGIN,
        budget_end = module_abi::ALLOCATION_BUDGET_END,
        set_publication = module_abi::SET_PUBLICATION_CALLBACK,
        error_type = dialect.error_type(),
        device_alloc = dialect.device_alloc_fn(),
        success = dialect.success_value(),
        error_string = dialect.error_string_fn(),
    )
}
