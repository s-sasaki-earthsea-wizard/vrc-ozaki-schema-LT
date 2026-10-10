! Plain Fortran DGEMM driver: C = A * B through the standard BLAS interface only.
!
! This stands in for an existing Fortran code that nobody wants to touch. It is
! compiled once against OpenBLAS; the same binary then runs
!   - on the CPU (OpenBLAS), or
!   - on the GPU by preloading NVBLAS (LD_PRELOAD=libnvblas.so), which intercepts
!     dgemm_ and forwards it to cuBLAS, where CUBLAS_EMULATE_DOUBLE_PRECISION
!     selects native FP64 or the Ozaki-scheme emulation.
! With NVBLAS the timing is end to end: host -> device copies, GEMM, device -> host.
!
! Inputs follow benchmark_dgemm.py: A = (U(0,1) - 0.5) * exp(phi * N(0,1)), but
! with the Fortran RNG, so the matrices are not bitwise identical to the Python ones.
! Accuracy is checked on sampled entries against a quad-precision (real128) dot
! product; products of two doubles are exact in real128, so the reference error is
! far below the FP64 unit roundoff u = 2^-53.
!
! Usage: dgemm_bench N [PHI=0.5] [REPEATS=5] [SAMPLES=64] [SEED=20261006]
program dgemm_bench
    use, intrinsic :: iso_fortran_env, only: int64, real64, real128
    implicit none

    external :: dgemm

    real(real64), parameter :: UNIT_ROUNDOFF = 2.0_real64**(-53)
    real(real64), parameter :: PI = 3.14159265358979323846_real64

    integer :: n, repeats, samples, seed, r
    real(real64) :: phi, warmup_s, flops, med_s
    real(real64) :: max_err_u, mean_err_u
    real(real64), allocatable :: a(:, :), b(:, :), c(:, :), times(:)
    integer(int64) :: t0, t1, rate

    n = int_arg(1, -1)
    if (n <= 0) then
        write (*, '(a)') 'usage: dgemm_bench N [PHI] [REPEATS] [SAMPLES] [SEED]'
        stop 2
    end if
    phi = real_arg(2, 0.5_real64)
    repeats = max(1, int_arg(3, 5))
    samples = max(0, int_arg(4, 64))
    seed = int_arg(5, 20261006)

    call seed_rng(seed)
    allocate (a(n, n), b(n, n), c(n, n), times(repeats))
    call make_matrix(a, phi)
    call make_matrix(b, phi)
    c = 0.0_real64

    call system_clock(count_rate=rate)

    ! The first call also pays for library initialisation (cuBLAS handle, heuristics).
    call system_clock(t0)
    call dgemm('N', 'N', n, n, n, 1.0_real64, a, n, b, n, 0.0_real64, c, n)
    call system_clock(t1)
    warmup_s = real(t1 - t0, real64) / real(rate, real64)

    do r = 1, repeats
        call system_clock(t0)
        call dgemm('N', 'N', n, n, n, 1.0_real64, a, n, b, n, 0.0_real64, c, n)
        call system_clock(t1)
        times(r) = real(t1 - t0, real64) / real(rate, real64)
    end do

    flops = 2.0_real64 * real(n, real64)**3
    med_s = median(times)
    call sampled_errors(a, b, c, samples, max_err_u, mean_err_u)

    write (*, '(a)') 'RESULT n=' // fixed(real(n, real64), 0) // ' phi=' // fixed(phi, 2) // &
        ' repeats=' // fixed(real(repeats, real64), 0) // ' samples=' // fixed(real(samples, real64), 0) // &
        ' warmup_s=' // fixed(warmup_s, 4) // ' median_s=' // fixed(med_s, 4) // &
        ' min_s=' // fixed(minval(times), 4) // ' gflops_median=' // fixed(flops / med_s / 1.0e9_real64, 1) // &
        ' max_scaled_err_u=' // fixed(max_err_u, 3) // ' mean_scaled_err_u=' // fixed(mean_err_u, 3)

contains

    ! Fixed-point text with a leading zero (the F0.d edit descriptor drops it).
    function fixed(x, digits) result(s)
        real(real64), intent(in) :: x
        integer, intent(in) :: digits
        character(len=:), allocatable :: s
        character(len=40) :: buf, fmt

        if (digits == 0) then
            write (buf, '(i0)') nint(x, int64)
        else
            write (fmt, '(a,i0,a)') '(f40.', digits, ')'
            write (buf, fmt) x
        end if
        s = trim(adjustl(buf))
    end function fixed

    integer function int_arg(pos, default) result(val)
        integer, intent(in) :: pos, default
        character(len=64) :: buf

        val = default
        if (command_argument_count() >= pos) then
            call get_command_argument(pos, buf)
            read (buf, *) val
        end if
    end function int_arg

    real(real64) function real_arg(pos, default) result(val)
        integer, intent(in) :: pos
        real(real64), intent(in) :: default
        character(len=64) :: buf

        val = default
        if (command_argument_count() >= pos) then
            call get_command_argument(pos, buf)
            read (buf, *) val
        end if
    end function real_arg

    subroutine seed_rng(s)
        integer, intent(in) :: s
        integer :: k, i
        integer, allocatable :: put(:)

        call random_seed(size=k)
        allocate (put(k))
        put = [(s + 7919 * i, i = 1, k)]
        call random_seed(put=put)
    end subroutine seed_rng

    ! x = (U(0,1) - 0.5) * exp(phi * N(0,1)), filled column by column to keep
    ! the temporaries at O(n) memory. N(0,1) comes from Box-Muller.
    subroutine make_matrix(x, phi_)
        real(real64), intent(out) :: x(:, :)
        real(real64), intent(in) :: phi_
        real(real64), allocatable :: u(:), v(:), w(:)
        integer :: j, m

        m = size(x, 1)
        allocate (u(m), v(m), w(m))
        do j = 1, size(x, 2)
            call random_number(u)
            x(:, j) = u - 0.5_real64
            if (phi_ /= 0.0_real64) then
                call random_number(v)
                call random_number(w)
                x(:, j) = x(:, j) * exp(phi_ * sqrt(-2.0_real64 * log(1.0_real64 - v)) * cos(2.0_real64 * PI * w))
            end if
        end do
    end subroutine make_matrix

    real(real64) function median(t) result(med)
        real(real64), intent(in) :: t(:)
        real(real64) :: s(size(t)), key
        integer :: i, j, m

        s = t
        m = size(s)
        do i = 2, m
            key = s(i)
            j = i - 1
            do while (j >= 1)
                if (s(j) <= key) exit
                s(j + 1) = s(j)
                j = j - 1
            end do
            s(j + 1) = key
        end do
        if (mod(m, 2) == 1) then
            med = s((m + 1) / 2)
        else
            med = 0.5_real64 * (s(m / 2) + s(m / 2 + 1))
        end if
    end function median

    ! Componentwise error |c_ij - exact_ij| / (|A||B|)_ij in units of u, as in
    ! benchmark_dgemm.py, on randomly sampled entries.
    subroutine sampled_errors(a_, b_, c_, ns, max_u, mean_u)
        real(real64), intent(in) :: a_(:, :), b_(:, :), c_(:, :)
        integer, intent(in) :: ns
        real(real64), intent(out) :: max_u, mean_u
        real(real128) :: acc, scale, p
        real(real64) :: rnd(2), e
        integer :: k, i, j, l, m

        max_u = 0.0_real64
        mean_u = 0.0_real64
        if (ns == 0) return
        m = size(a_, 1)
        do k = 1, ns
            call random_number(rnd)
            i = min(m, 1 + int(rnd(1) * m))
            j = min(m, 1 + int(rnd(2) * m))
            acc = 0.0_real128
            scale = 0.0_real128
            do l = 1, m
                p = real(a_(i, l), real128) * real(b_(l, j), real128)
                acc = acc + p
                scale = scale + abs(p)
            end do
            e = real(abs(real(c_(i, j), real128) - acc) / scale, real64) / UNIT_ROUNDOFF
            max_u = max(max_u, e)
            mean_u = mean_u + e
        end do
        mean_u = mean_u / ns
    end subroutine sampled_errors

end program dgemm_bench
