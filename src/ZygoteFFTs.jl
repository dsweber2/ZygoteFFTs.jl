module ZygoteFFTs
using Zygote
using AbstractFFTs
using FillArrays

import Zygote:@adjoint

AbstractFFTs.bfft(x::Fill, dims...) = AbstractFFTs.bfft(collect(x), dims...)
AbstractFFTs.rfft(x::Fill, dims...) = AbstractFFTs.rfft(collect(x), dims...)
AbstractFFTs.irfft(x::Fill, d, dims...) = AbstractFFTs.irfft(collect(x), d, dims...)

@adjoint function fft(xs)
    return AbstractFFTs.fft(xs), function(Δ)
        return (AbstractFFTs.bfft(Δ),)
    end
end

# all of the plans normalize their inverse, while we need the unnormalized one.
@adjoint function *(P::AbstractFFTs.Plan, xs)
    return P * xs, function(Δ)
        N = prod(size(xs)[[P.region...]])
        return (nothing, N * (P \ Δ))
    end
end

@adjoint function \(P::AbstractFFTs.Plan, xs)
    return P \ xs, function(Δ)
        N = length(Δ)
        return (nothing, 1/N * (P * Δ))
    end
end

@adjoint function ifft(xs)
    return AbstractFFTs.ifft(xs), function(Δ)
        N = length(xs)
        return (1/N* AbstractFFTs.fft(Δ),)
    end
end

@adjoint function bfft(xs)
    return AbstractFFTs.bfft(xs), function(Δ)
        return (AbstractFFTs.fft(Δ),)
    end
end

# to actually use rfft, one needs to insure that everything that happens in the
# Fourier domain could've been done in the space domain with real numbers. This
# means enforcing conjugate symmetry along all transformed dimensions besides
# the first. Otherwise this is going to result in *very* weird behavior. 
@adjoint function rfft(xs::AbstractArray{<:Real})
    return AbstractFFTs.rfft(xs), function(Δ)
        N = length(Δ)
        originalSize = size(xs,1)
        return (AbstractFFTs.brfft(Δ, originalSize),)
    end
end

@adjoint function irfft(xs, d)
    return AbstractFFTs.irfft(xs, d), function(Δ)
        total = length(Δ)
        fullTransform = 1/total * AbstractFFTs.rfft(real.(Δ))
        return (fullTransform, nothing)
    end
end

@adjoint function brfft(xs, d)
    return AbstractFFTs.brfft(xs, d), function(Δ)
        fullTransform = AbstractFFTs.rfft(real.(Δ))
        return (fullTransform, nothing)
    end
end

@adjoint function fftshift(x)
    return fftshift(x), function(Δ)
        return (ifftshift(Δ),)
    end
end

@adjoint function ifftshift(x)
    return ifftshift(x), function(Δ)
        return (fftshift(Δ),)
    end
end

# if we're specifying the dimensions
@adjoint function fft(xs, dims)
    return AbstractFFTs.fft(xs, dims), function(Δ)
        # dims can be int, array or tuple,
        # convert to collection for use as index
        dims = collect(dims)
        return (AbstractFFTs.bfft(Δ, dims), nothing)
    end
end

@adjoint function bfft(xs, dims)
    return AbstractFFTs.ifft(xs, dims), function(Δ)
        dims = collect(dims)
        return (AbstractFFTs.fft(Δ, dims),nothing)
    end
end

@adjoint function ifft(xs, dims)
    return AbstractFFTs.ifft(xs, dims), function(Δ)
        dims = collect(dims)
        N = prod(collect(size(xs))[dims])
        return (1/N * AbstractFFTs.fft(Δ, dims),nothing)
    end
end


@adjoint function rfft(xs, dims)
    return AbstractFFTs.rfft(xs, dims), function(Δ)
        dims = collect(dims)
        N = prod(collect(size(xs))[dims])
        return (N * AbstractFFTs.irfft(Δ, size(xs,dims[1]), dims), nothing)
    end
end


@adjoint function irfft(xs, d, dims)
    return AbstractFFTs.ifft(xs, dims), function(Δ)
        dims = collect(dims)
        N = prod(collect(size(xs))[dims])
        return (1/N * AbstractFFTs.rfft(Δ, dims), nothing, nothing)
    end
end
@adjoint function brfft(xs, d, dims)
    return AbstractFFTs.ifft(xs, dims), function(Δ)
        dims = collect(dims)
        return (AbstractFFTs.rfft(Δ, dims), nothing, nothing)
    end
end


@adjoint function fftshift(x, dims)
    return fftshift(x), function(Δ)
        return (ifftshift(Δ, dims), nothing)
    end
end

@adjoint function ifftshift(x, dims)
    return ifftshift(x), function(Δ)
        return (fftshift(Δ, dims), nothing)
    end
end

end # module
