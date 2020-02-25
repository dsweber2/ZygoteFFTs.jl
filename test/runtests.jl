using ZygoteFFTs
using Test
using FillArrays
using FFTW
using Zygote

function ngradient(f, xs::AbstractArray...)
  grads = zero.(xs)
  for (x, Δ) in zip(xs, grads), i in 1:length(x)
    δ = sqrt(eps())
    tmp = x[i]
    x[i] = tmp - δ/2
    y1 = f(xs...)
    x[i] = tmp + δ/2
    y2 = f(xs...)
    x[i] = tmp
    Δ[i] = (y2-y1)/δ
  end
  return grads
end

gradcheck(f, xs...) =
  all(isapprox.(ngradient(f, xs...),
                gradient(f, xs...), rtol = 1e-5, atol = 1e-5))

gradtest(f, xs::AbstractArray...) = gradcheck((xs...) -> sum(sin.(f(xs...))), xs...)
gradtest(f, dims...) = gradtest(f, rand.(Float64, dims)...)

# utilities for using gradcheck with complex matrices
_splitreim(A) = (real(A),)
_splitreim(A::AbstractArray{<:Complex}) = reim(A)

_joinreim(A, B) = complex.(A, B)
_joinreim(A) = A

function _dropimaggrad(A)
  back(Δ) = real(Δ)
  back(Δ::Nothing) = nothing
  return Zygote.hook(back, A)
end


@testset "ZygoteFFTs.jl" begin
    findicateMat(i,j,n1,n2) = [(k==i) && (l==j) ? 1.0 : 0.0 for k=1:n1,
                               l=1:n2]
    mirrorIndex(i,N) = i - 2*max(0,i - (N>>1+1))
    for sizeX in [(2,3), (10,10), (13,15)]
        X = randn(sizeX)
        X̂r = rfft(X)
        X̂ = fft(X)
        N = prod(sizeX)
        for i=1:size(X,1), j=1:size(X,2)
            indicateMat = [(k==i) && (l==j) ? 1.0 : 0.0 for k=1:size(X, 1),
                           l=1:size(X,2)]
            # gradient of ifft(fft) must be (approximately) 1 (for various cases)
            @test gradient((X)->real.(ifft(fft(X))[i, j]), X)[1] ≈ indicateMat
            # same for the inverse
            @test gradient((X̂)->real.(fft(ifft(X̂))[i, j]), X̂)[1] ≈ indicateMat
            # same for rfft(irfft)
            @test gradient((X)->real.(irfft(rfft(X), 
                                            size(X,1)))[i, j], X)[1] ≈
                                                real.(indicateMat)
            # rfft isn't actually surjective, so rffft(irfft) can't really
            # be tested this way. 
            
            # the gradients are actually just evaluating the inverse
            # transform on the indicator matrix
            mirrorI = mirrorIndex(i,sizeX[1])
            FreqIndMat = findicateMat(mirrorI, j, size(X̂r,1), sizeX[2])
            listOfSols = [(fft, bfft(indicateMat), bfft(indicateMat*im),
                           plan_fft(X), i, X),
                          (ifft, 1/N*fft(indicateMat), 1/N*fft(indicateMat*im),
                           plan_fft(X), i, X),
                          (bfft, fft(indicateMat), fft(indicateMat*im),
                           nothing, i, X),
                          (rfft, real.(brfft(FreqIndMat, sizeX[1])),
                           real.(brfft(FreqIndMat*im, sizeX[1])), plan_rfft(X),
                           mirrorI, X),
                          ((K)->(irfft(K,sizeX[1])), 1/N * rfft(indicateMat),
                           zeros(size(X̂r)), plan_rfft(X), i, X̂r)]
            for (trans, solRe, solIm, P, mI, evalX) in listOfSols
                @test gradient((X)->real.(trans(X))[mI, j], evalX)[1] ≈
                    solRe
                @test gradient((X)->imag.(trans(X))[mI, j], evalX)[1] ≈
                    solIm
                if typeof(P) <:AbstractFFTs.Plan && maximum(trans .== [fft,rfft])
                    @test gradient((X)->real.(P * X)[mI, j], evalX)[1] ≈
                        solRe
                    @test gradient((X)->imag.(P * X)[mI, j], evalX)[1] ≈
                        solIm
                elseif typeof(P) <: AbstractFFTs.Plan
                    @test gradient((X)->real.(P \ X)[mI, j], evalX)[1] ≈
                        solRe
                    # for whatever reason the rfft_plan doesn't handle this
                    # case well, even though irfft does
                    if eltype(evalX) <: Real
                        @test gradient((X)->imag.(P \ X)[mI, j], evalX)[1] ≈
                            solIm
                    end
                end
            end
        end
    end
    
    x = [-0.353213 -0.789656 -0.270151; -0.95719 -1.27933 0.223982]  
    for trans in (fft, ifft, bfft)
        @test gradient((x)->sum(abs.(trans(x))), x)[1] ≈
            gradient( (x) -> sum(abs.(trans(trans(x,1),2))),  x)[1]
        # switch sum abs order
        @test gradient((x)->abs(sum((trans(x)))),x)[1] ≈
            gradient( (x) -> abs(sum(trans(trans(x,1),2))),  x)[1]
        # dims parameter for the function
        @test gradient((x, dims)->sum(abs.(trans(x,dims))), x, (1,2))[1] ≈
            gradient( (x) -> sum(abs.(trans(x))), x)[1]
        # (1,2) should be the same as no index
        @test gradient( (x) -> sum(abs.(trans(x,(1,2)))), x)[1] ≈
            gradient( (x) -> sum(abs.(trans(trans(x,1),2))), x)[1]
        @test gradcheck(x->sum(abs.(trans(x))), x)
        @test gradcheck(x->sum(abs.(trans(x, 2))), x)
    end
    
    @test gradient((x)->sum(abs.(rfft(x))), x)[1] ≈
        gradient( (x) -> sum(abs.(fft(rfft(x,1),2))),  x)[1]
    @test gradient((x, dims)->sum(abs.(rfft(x,dims))), x, (1,2))[1] ≈
        gradient( (x) -> sum(abs.(rfft(x))), x)[1]
end
