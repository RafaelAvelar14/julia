# This file is a part of Julia. License is MIT: https://julialang.org/license

using Test, Base.Threads, InteractiveUtils

function addmodify!(f::Atomic{Int64}, v::Integer, ordering::Symbol = :sequentially_consistent)
    modifyfield!(f, :value, +, Int64(v), ordering)
end

function submodify!(f::Atomic{Int64}, v::Integer, ordering::Symbol = :sequentially_consistent)
    modifyfield!(f, :value, -, Int64(v), ordering)
end

function andmodify!(f::Atomic{Int64}, v::Integer, ordering::Symbol = :sequentially_consistent)
    modifyfield!(f, :value, &, Int64(v), ordering)
end

function ormodify!(f::Atomic{Int64}, v::Integer, ordering::Symbol = :sequentially_consistent)
    modifyfield!(f, :value, |, Int64(v), ordering)
end

function xormodify!(f::Atomic{Int64}, v::Integer, ordering::Symbol = :sequentially_consistent)
    modifyfield!(f, :value, ⊻, Int64(v), ordering)
end

function mulmodify!(f::Atomic{Int64}, v::Integer, ordering::Symbol = :sequentially_consistent)
    modifyfield!(f, :value, *, Int64(v), ordering)
end

@testset "Atomic modifyfield! operations" begin
    f = Atomic{Int64}(0)

    addmodify!(f, 10)
    @test f.value == 10   # 0 + 10 == 10

    submodify!(f, 3)
    @test f.value == 7    # 10 − 3 == 7

    mulmodify!(f, 4)
    @test f.value == 28   # 7 * 4 == 28

    andmodify!(f, 0xF)
    @test f.value == (28 & 0xF)   # 28 & 0xF == 12

    ormodify!(f, 0x10)
    @test f.value == (12 | 0x10)   # 12 | 0x10 == 28

    xormodify!(f, 0x3)
    @test f.value == (28 ⊻ 0x3)    # 28 ⊻ 3 == 31

    let
        ir_add = sprint(io ->
            InteractiveUtils.code_llvm(io, addmodify!, Tuple{Atomic{Int64}, Int64}; debuginfo=:none)
        )
        @test occursin("atomicrmw add", ir_add)
    end

    let
        ir_sub = sprint(io ->
            InteractiveUtils.code_llvm(io, submodify!, Tuple{Atomic{Int64}, Int64}; debuginfo=:none)
        )
        @test occursin("atomicrmw sub", ir_sub)
    end

    let
        ir_and = sprint(io ->
            InteractiveUtils.code_llvm(io, andmodify!, Tuple{Atomic{Int64}, Int64}; debuginfo=:none)
        )
        @test occursin("atomicrmw and", ir_and)
    end

    let
        ir_or = sprint(io ->
            InteractiveUtils.code_llvm(io, ormodify!, Tuple{Atomic{Int64}, Int64}; debuginfo=:none)
        )
        @test occursin("atomicrmw or", ir_or)
    end

    let
        ir_xor = sprint(io ->
            InteractiveUtils.code_llvm(io, xormodify!, Tuple{Atomic{Int64}, Int64}; debuginfo=:none)
        )
        @test occursin("atomicrmw xor", ir_xor)
    end

    let
        # LLVM does not provide a direct “atomicrmw mul” opcode, so mulmodify! should fallback to cmpxchg
        ir_mul = sprint(io ->
            InteractiveUtils.code_llvm(io, mulmodify!, Tuple{Atomic{Int64}, Int64}; debuginfo=:none)
        )
        @test occursin("cmpxchg", ir_mul)
    end
end

@testset "Atomic modifyfield! operations with multi-threaded execution" begin
    NTHREADS = Threads.nthreads()
    NITER    = 100_000

    f = Atomic{Int64}(0)
    @sync for i in 1:NTHREADS
        @spawn begin
            for j in 1:NITER
                addmodify!(f, 1)
            end
        end
    end

    @test f.value == NTHREADS * NITER
end

@testset "Atomic modifyfield! operations with ordering" begin
    f = Atomic{Int64}(0)

    # addmodify!
    addmodify_seq_cst!(f, v) = addmodify!(f, v, :sequentially_consistent)
    addmodify_mon!(f, v) = addmodify!(f, v, :monotonic)
    addmodify_acq!(f, v) = addmodify!(f, v, :acquire)
    addmodify_rel!(f, v) = addmodify!(f, v, :release)
    addmodify_acq_rel!(f, v) = addmodify!(f, v, :acquire_release)

    ir_add_seq_cst = sprint(io -> InteractiveUtils.code_llvm(io, addmodify_seq_cst!, Tuple{Atomic{Int64}, Int64}; debuginfo=:none))
    ir_add_mon = sprint(io -> InteractiveUtils.code_llvm(io, addmodify_mon!, Tuple{Atomic{Int64}, Int64}; debuginfo=:none))
    ir_add_acq = sprint(io -> InteractiveUtils.code_llvm(io, addmodify_acq!, Tuple{Atomic{Int64}, Int64}; debuginfo=:none))
    ir_add_rel = sprint(io -> InteractiveUtils.code_llvm(io, addmodify_rel!, Tuple{Atomic{Int64}, Int64}; debuginfo=:none))
    ir_add_acq_rel = sprint(io -> InteractiveUtils.code_llvm(io, addmodify_acq_rel!, Tuple{Atomic{Int64}, Int64}; debuginfo=:none))

    @test occursin("seq_cst", ir_add_seq_cst)
    @test occursin("monotonic", ir_add_mon)
    @test occursin("acquire", ir_add_acq)
    @test occursin("release", ir_add_rel)
    @test occursin("acq_rel", ir_add_acq_rel)

    # submodify!
    submodify_seq_cst!(f, v) = submodify!(f, v, :sequentially_consistent)
    submodify_mon!(f, v) = submodify!(f, v, :monotonic)
    submodify_acq!(f, v) = submodify!(f, v, :acquire)
    submodify_rel!(f, v) = submodify!(f, v, :release)
    submodify_acq_rel!(f, v) = submodify!(f, v, :acquire_release)

    ir_sub_seq_cst = sprint(io -> InteractiveUtils.code_llvm(io, submodify_seq_cst!, Tuple{Atomic{Int64}, Int64}; debuginfo=:none))
    ir_sub_mon = sprint(io -> InteractiveUtils.code_llvm(io, submodify_mon!, Tuple{Atomic{Int64}, Int64}; debuginfo=:none))
    ir_sub_acq = sprint(io -> InteractiveUtils.code_llvm(io, submodify_acq!, Tuple{Atomic{Int64}, Int64}; debuginfo=:none))
    ir_sub_rel = sprint(io -> InteractiveUtils.code_llvm(io, submodify_rel!, Tuple{Atomic{Int64}, Int64}; debuginfo=:none))
    ir_sub_acq_rel = sprint(io -> InteractiveUtils.code_llvm(io, submodify_acq_rel!, Tuple{Atomic{Int64}, Int64}; debuginfo=:none))

    @test occursin("seq_cst", ir_sub_seq_cst)
    @test occursin("monotonic", ir_sub_mon)
    @test occursin("acquire", ir_sub_acq)
    @test occursin("release", ir_sub_rel)
    @test occursin("acq_rel", ir_sub_acq_rel)

    # andmodify!
    andmodify_seq_cst!(f, v) = andmodify!(f, v, :sequentially_consistent)
    andmodify_mon!(f, v) = andmodify!(f, v, :monotonic)
    andmodify_acq!(f, v) = andmodify!(f, v, :acquire)
    andmodify_rel!(f, v) = andmodify!(f, v, :release)
    andmodify_acq_rel!(f, v) = andmodify!(f, v, :acquire_release)

    ir_and_seq_cst = sprint(io -> InteractiveUtils.code_llvm(io, andmodify_seq_cst!, Tuple{Atomic{Int64}, Int64}; debuginfo=:none))
    ir_and_mon = sprint(io -> InteractiveUtils.code_llvm(io, andmodify_mon!, Tuple{Atomic{Int64}, Int64}; debuginfo=:none))
    ir_and_acq = sprint(io -> InteractiveUtils.code_llvm(io, andmodify_acq!, Tuple{Atomic{Int64}, Int64}; debuginfo=:none))
    ir_and_rel = sprint(io -> InteractiveUtils.code_llvm(io, andmodify_rel!, Tuple{Atomic{Int64}, Int64}; debuginfo=:none))
    ir_and_acq_rel = sprint(io -> InteractiveUtils.code_llvm(io, andmodify_acq_rel!, Tuple{Atomic{Int64}, Int64}; debuginfo=:none))

    @test occursin("seq_cst", ir_and_seq_cst)
    @test occursin("monotonic", ir_and_mon)
    @test occursin("acquire", ir_and_acq)
    @test occursin("release", ir_and_rel)
    @test occursin("acq_rel", ir_and_acq_rel)

    # ormodify!
    ormodify_seq_cst!(f, v) = ormodify!(f, v, :sequentially_consistent)
    ormodify_mon!(f, v) = ormodify!(f, v, :monotonic)
    ormodify_acq!(f, v) = ormodify!(f, v, :acquire)
    ormodify_rel!(f, v) = ormodify!(f, v, :release)
    ormodify_acq_rel!(f, v) = ormodify!(f, v, :acquire_release)

    ir_or_seq_cst = sprint(io -> InteractiveUtils.code_llvm(io, ormodify_seq_cst!, Tuple{Atomic{Int64}, Int64}; debuginfo=:none))
    ir_or_mon = sprint(io -> InteractiveUtils.code_llvm(io, ormodify_mon!, Tuple{Atomic{Int64}, Int64}; debuginfo=:none))
    ir_or_acq = sprint(io -> InteractiveUtils.code_llvm(io, ormodify_acq!, Tuple{Atomic{Int64}, Int64}; debuginfo=:none))
    ir_or_rel = sprint(io -> InteractiveUtils.code_llvm(io, ormodify_rel!, Tuple{Atomic{Int64}, Int64}; debuginfo=:none))
    ir_or_acq_rel = sprint(io -> InteractiveUtils.code_llvm(io, ormodify_acq_rel!, Tuple{Atomic{Int64}, Int64}; debuginfo=:none))

    @test occursin("seq_cst", ir_or_seq_cst)
    @test occursin("monotonic", ir_or_mon)
    @test occursin("acquire", ir_or_acq)
    @test occursin("release", ir_or_rel)
    @test occursin("acq_rel", ir_or_acq_rel)

    # xormodify!

    xormodify_seq_cst!(f, v) = xormodify!(f, v, :sequentially_consistent)
    xormodify_mon!(f, v) = xormodify!(f, v, :monotonic)
    xormodify_acq!(f, v) = xormodify!(f, v, :acquire)
    xormodify_rel!(f, v) = xormodify!(f, v, :release)
    xormodify_acq_rel!(f, v) = xormodify!(f, v, :acquire_release)

    ir_xor_seq_cst = sprint(io -> InteractiveUtils.code_llvm(io, xormodify_seq_cst!, Tuple{Atomic{Int64}, Int64}; debuginfo=:none))
    ir_xor_mon = sprint(io -> InteractiveUtils.code_llvm(io, xormodify_mon!, Tuple{Atomic{Int64}, Int64}; debuginfo=:none))
    ir_xor_acq = sprint(io -> InteractiveUtils.code_llvm(io, xormodify_acq!, Tuple{Atomic{Int64}, Int64}; debuginfo=:none))
    ir_xor_rel = sprint(io -> InteractiveUtils.code_llvm(io, xormodify_rel!, Tuple{Atomic{Int64}, Int64}; debuginfo=:none))
    ir_xor_acq_rel = sprint(io -> InteractiveUtils.code_llvm(io, xormodify_acq_rel!, Tuple{Atomic{Int64}, Int64}; debuginfo=:none))

    @test occursin("seq_cst", ir_xor_seq_cst)
    @test occursin("monotonic", ir_xor_mon)
    @test occursin("acquire", ir_xor_acq)
    @test occursin("release", ir_xor_rel)
    @test occursin("acq_rel", ir_xor_acq_rel)

end
