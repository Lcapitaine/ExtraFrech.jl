using ExtraFrech
using Test

@testset "ExtraFrech.jl" begin
    # Write your tests here.
    @test ExtraFrech.impurity(zeros(10)) == 0
    
end
