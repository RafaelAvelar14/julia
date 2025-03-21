// This file is a part of Julia. License is MIT: https://julialang.org/license

#ifndef JL_GCEXT_H
#define JL_GCEXT_H

#ifdef __cplusplus
extern "C" {
#endif

// requires including "julia.h" beforehand.

// Callbacks that allow C code to hook into the GC.

// Marking callbacks for global roots and tasks, respectively. These,
// along with custom mark functions must not alter the GC state except
// through calling jl_gc_mark_queue_obj() and jl_gc_mark_queue_objarray().
typedef void (*jl_gc_cb_root_scanner_t)(int full) JL_NOTSAFEPOINT;
typedef void (*jl_gc_cb_task_scanner_t)(jl_task_t *task, int full) JL_NOTSAFEPOINT;

// Callbacks that are invoked before and after a collection.
typedef void (*jl_gc_cb_pre_gc_t)(int full) JL_NOTSAFEPOINT;
typedef void (*jl_gc_cb_post_gc_t)(int full) JL_NOTSAFEPOINT;

// Callbacks to track external object allocations.
typedef void (*jl_gc_cb_notify_external_alloc_t)(void *addr, size_t size) JL_NOTSAFEPOINT;
typedef void (*jl_gc_cb_notify_external_free_t)(void *addr) JL_NOTSAFEPOINT;

JL_DLLEXPORT void jl_gc_set_cb_root_scanner(jl_gc_cb_root_scanner_t cb, int enable);
JL_DLLEXPORT void jl_gc_set_cb_task_scanner(jl_gc_cb_task_scanner_t cb, int enable);
JL_DLLEXPORT void jl_gc_set_cb_pre_gc(jl_gc_cb_pre_gc_t cb, int enable);
JL_DLLEXPORT void jl_gc_set_cb_post_gc(jl_gc_cb_post_gc_t cb, int enable);
JL_DLLEXPORT void jl_gc_set_cb_notify_external_alloc(jl_gc_cb_notify_external_alloc_t cb,
        int enable);
JL_DLLEXPORT void jl_gc_set_cb_notify_external_free(jl_gc_cb_notify_external_free_t cb,
        int enable);

// Memory pressure callback
typedef void (*jl_gc_cb_notify_gc_pressure_t)(void) JL_NOTSAFEPOINT;
JL_DLLEXPORT void jl_gc_set_cb_notify_gc_pressure(jl_gc_cb_notify_gc_pressure_t cb, int enable);

// Types for custom mark and sweep functions.
typedef uintptr_t (*jl_markfunc_t)(jl_ptls_t, jl_value_t *obj) JL_NOTSAFEPOINT;
typedef void (*jl_sweepfunc_t)(jl_value_t *obj) JL_NOTSAFEPOINT;

// Function to create a new foreign type with custom
// mark and sweep functions.
JL_DLLEXPORT jl_datatype_t *jl_new_foreign_type(
        jl_sym_t *name,
        jl_module_t *module,
        jl_datatype_t *super,
        jl_markfunc_t markfunc,
        jl_sweepfunc_t sweepfunc,
        int haspointers,
        int large);


#define HAVE_JL_REINIT_FOREIGN_TYPE 1
JL_DLLEXPORT int jl_reinit_foreign_type(
        jl_datatype_t *dt,
        jl_markfunc_t markfunc,
        jl_sweepfunc_t sweepfunc);

JL_DLLEXPORT int jl_is_foreign_type(jl_datatype_t *dt) JL_NOTSAFEPOINT;

JL_DLLEXPORT size_t jl_gc_max_internal_obj_size(void) JL_NOTSAFEPOINT;
JL_DLLEXPORT size_t jl_gc_external_obj_hdr_size(void) JL_NOTSAFEPOINT;

// Field layout descriptor for custom types that do
// not fit Julia layout conventions. This is associated with
// jl_datatype_t instances where fielddesc_type == 3.

typedef struct {
    jl_markfunc_t markfunc;
    jl_sweepfunc_t sweepfunc;
} jl_fielddescdyn_t;

// Allocate an object of a foreign type.
JL_DLLEXPORT void *jl_gc_alloc_typed(jl_ptls_t ptls, size_t sz, void *ty);

// Queue an object or array of objects for scanning by the garbage collector.
// These functions must only be called from within a root scanner callback
// or from within a custom mark function.
JL_DLLEXPORT int jl_gc_mark_queue_obj(jl_ptls_t ptls, jl_value_t *obj) JL_NOTSAFEPOINT;
JL_DLLEXPORT void jl_gc_mark_queue_objarray(jl_ptls_t ptls, jl_value_t *parent,
    jl_value_t **objs, size_t nobjs) JL_NOTSAFEPOINT;

// Sweep functions will not automatically be called for objects of
// foreign types, as that may not always be desired. Only calling
// jl_gc_schedule_foreign_sweepfunc() on an object of a foreign type
// will result in the custom sweep function actually being called.
// This must be done at most once per object and should usually be
// done right after allocating the object.
JL_DLLEXPORT void jl_gc_schedule_foreign_sweepfunc(jl_ptls_t ptls, jl_value_t *bj);

// The following functions enable support for conservative marking. This
// functionality allows the user to determine if a machine word can be
// interpreted as a pointer to an object (including the interior of an
// object). It can be used to, for example, scan foreign stack frames or
// data structures with an unknown layout. It is called conservative
// marking, because it can lead to false positives, as non-pointer data
// can mistakenly be interpreted as a pointer to a Julia object.

// CAUTION: This is a sharp tool and should only be used as a measure of
// last resort. The user should be familiar with the risk of memory
// leaks (especially on 32-bit machines) if used inappropriately and how
// optimizing compilers can hide references from conservative stack
// scanning. In particular, arrays must be kept explicitly visible to
// the GC (by using JL_GC_PUSH1(), storing them in a location marked by
// the Julia GC, etc.) while their contents are being accessed. The
// reason is that array contents aren't marked separately, so if the
// object itself is not visible to the GC, neither are the contents.

// Enable support for conservative marking. The function returns
// whether support was already enabled. The function may implicitly
// trigger a full garbage collection to properly update all internal
// data structures.
JL_DLLEXPORT int jl_gc_enable_conservative_gc_support(void);

// This function returns whether support for conservative scanning has
// been enabled. The return values are the same as for
// jl_gc_enable_conservative_gc_support().
JL_DLLEXPORT int jl_gc_conservative_gc_support_enabled(void);

// Returns the base address of a memory block, assuming it is stored in
// a julia memory pool. Return NULL otherwise. Conservative support
// *must* have been enabled for this to work reliably.
//
// NOTE: This will only work for internal pool allocations. For external
// allocations, the user must track allocations using the notification
// callbacks above and verify that they are valid objects. Note that
// external allocations may not all be valid objects and that for those,
// the user *must* validate that they have a proper type, i.e. that
// jl_typeof(obj) is an actual type object.
//
// NOTE: Only valid to call from within a GC context.
JL_DLLEXPORT jl_value_t *jl_gc_internal_obj_base_ptr(void *p) JL_NOTSAFEPOINT;

// Query the active and total stack range for the given task, and set
// *active_start and *active_end respectively *total_start and *total_end
// accordingly. The range for the active part is a best-effort approximation
// and may not be tight.
JL_DLLEXPORT void jl_active_task_stack(jl_task_t *task,
                                       char **active_start, char **active_end,
                                       char **total_start, char **total_end) JL_NOTSAFEPOINT;

#ifdef __cplusplus
}
#endif

#endif // _JULIA_GCEXT_H
