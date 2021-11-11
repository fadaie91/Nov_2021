ENV["GKSwstype"] = "nul"
using Printf
using Oceananigans
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Plots
using JLD2
using LinearAlgebra
using Oceananigans.ImmersedBoundaries: solid_node
using Oceananigans.ImmersedBoundaries: mask_immersed_field!

function show_mask(grid)

    print("grid = ", grid, "\n")
    c = CenterField(CPU(), grid)
    c .= 1

    mask_immersed_field!(c)

    x, y, z = nodes(c)

    return x, y, z, c
end

grid = RegularRectilinearGrid(size=(128, 64),
                              y=(-1, 1),                        
                              z=(-1, 0),                           
                              topology=(Flat, Periodic, Bounded))

# Gaussian seamount
h0, L = 0.5, 0.25                                        
seamount(x, y, z) = z < - 1 + h0*exp(-y^2/L^2)  
grid_with_seamount = ImmersedBoundaryGrid(grid, GridFittedBoundary(seamount))

#x, y, z, cc = show_mask(grid_with_seamount)
#plt = contourf(y, z, interior(cc)[1,:,:]', xlabel = "y", ylabel = "z", title = "Masked Region")
#savefig(plt,"mask.png")


### Here we define psi and then fill the halo reginos
Ψ = Field(Center, Face, Face, CPU(), grid_with_seamount)
h(y)    = h0*exp(-y^2/L^2)
ζ(y, z) = z/(h(y) - 1)
set!(Ψ, (x, y, z) -> (1 - ζ(y, z))^2)
fill_halo_regions!(Ψ, CPU())

### We mask psi
mask_immersed_field!(Ψ)

### V is (Center, Face, Center)
V = YFaceField(CPU(), grid_with_seamount)
### W is (Center, Centere, Face)
W = ZFaceField(CPU(), grid_with_seamount)

### V and W are deravatives of psi 
### which in this code we use exact expression 
### which we computed in maple
### we are calling this method 'analytical'
set!(V, (x, y, z) -> -2 * (1 - ( z / (-1+h0*exp(-y^2/L^2)) ) )/(-1+h0*exp(-y^2/L^2)))
set!(W, (x, y, z) -> 4*h0*z*y*exp(-y^2/L^2)*(1 - z/(-1 + h0*exp(-y^2/L^2)))/((L^2)*(-1 + h0*exp(-y^2/L^2))^2))

### We fill the halo regions of V and W
fill_halo_regions!(V, CPU())
fill_halo_regions!(W, CPU())

### We mask V and W        
mask_immersed_field!(V)
mask_immersed_field!(W)

### In this example of the paper the velocities were prescribe
### which means that they won't evolve during the simulation
### and using the function below we prescribe the velocities
velocities = PrescribedVelocityFields( v=V, w=W)

model = HydrostaticFreeSurfaceModel(architecture = CPU(),  
                                    momentum_advection = UpwindBiasedThirdOrder(),                         
                                    grid = grid_with_seamount,
                                    tracers = :θ,
                                   velocities = velocities,
                                    buoyancy = nothing
                                    )


### we define a Field 'B'  which is defining the tracer and then we masked it                                    
B = Field(Center, Center, Center, CPU(), grid_with_seamount)     
set!(B, (x, y, z) -> (1.0 + z) ) 
mask_immersed_field!(B)


Δt = 0.001/16

### since in this model only the tracer is changing 
### we should only put the initial value of tracer
### while setting the model (not the velocities)
set!(model, θ = B)                       

progress(s) = @info @sprintf("[%.2f%%], iteration: %d, time: %.3f, max|θ|: %.2e",
                             100 * s.model.clock.time / s.stop_time, s.model.clock.iteration,
                             s.model.clock.time, maximum(abs, model.tracers.θ))



### nodes of tracer are defined below
xθ, yθ, zθ = nodes((Center, Center, Center), grid)

### we plot θ before simulation
θplot = contourf(yθ, zθ, interior(model.tracers.θ)[1, :, :]', title="tracer", xlabel="y", ylabel="z")
savefig(θplot, "tracer_initial.png")

### here we define the total concentraction of tracer before simulation
tracer_initial_tot = sum(interior(model.tracers.θ))*1/64*1/64
G1 = Field(Center, Center, Center, CPU(), grid_with_seamount)
G1 .= ∂y(model.tracers.θ) 

G2 = Field(Center, Center, Center, CPU(), grid_with_seamount)
G2 .= ∂z(model.tracers.θ)

sqrt.(G1.^2 .+ G2.^2 )

simulation = Simulation(model, Δt = Δt, stop_time = 0.3, progress = progress, iteration_interval = 10)

serialize_grid(file, model) = file["serialized/grid"] = model.grid.grid
simulation.output_writers[:fields] = JLD2OutputWriter(model, model.tracers,
                                                      schedule = TimeInterval(0.02),
                                                      prefix = "flow_over_seamount",
                                                      init = serialize_grid,
                                                      force = true)
                        
start_time = time_ns()
run!(simulation)
finish_time = time_ns()

### we plot θ after simulation
θplot = contourf(yθ, zθ, interior(model.tracers.θ)[1, :, :]'; title="tracer", xlabel="y", ylabel="z", label=@sprintf("t = %.3f", model.clock.time))
savefig(θplot, "tracer_final.png")
### here we define the total concentraction of tracer after simulation
tracer_final_tot = sum(interior(model.tracers.θ))*1/64*1/64

G1 .= ∂y(model.tracers.θ) 
G2 .= ∂z(model.tracers.θ)
final_grad=sqrt.(G1.^2 .+ G2.^2 )
norm(final_grad, Inf)
grad_plot = contourf(yθ, zθ, (final_grad)[1, :, :]', title="final_grad", xlabel="y", ylabel="z")
savefig(grad_plot, "final_grad.png")
Ny = grid_with_seamount.grid.Ny
topography_index = zeros(Int64, Ny)
for (iyC, yC) in enumerate(grid_with_seamount.grid.yC[1:Ny])
    print("y = ", yC, " z = ", grid_with_seamount.grid.zC[1], " ")

    izC = 1
    test = true
    while test && izC <= grid_with_seamount.grid.Nz
    	  print(izC, " ")
	  test =  solid_node(Center(), Center(), Center(), 1, iyC, izC, grid_with_seamount)
	  izC += 1
    end
    print(" ")
    print(solid_node(Center(), Center(), Center(), 1,iyC,1, grid_with_seamount)," ")
    print(solid_node(Center(), Center(), Center(), 1,iyC,2, grid_with_seamount),"\n")
    topography_index[iyC] = izC - 1 
end

print(topography_index, "\n")

Theta_topography_slice = zeros(Ny)
for (iyC, yC) in enumerate(grid_with_seamount.grid.yC[1:Ny])
    Theta_topography_slice[iyC] = (model.tracers.θ)[1, iyC, topography_index[iyC]]
end
print(Theta_topography_slice, "\n")
y_slice=1:Ny
theta_slice_plot = plot(y_slice, Theta_topography_slice,  xlabel="y", ylabel="Theta_topography_slice")

savefig(theta_slice_plot, "theta_slice_plot.png")



@info """
    Simulation complete.
    Output: $(abspath(simulation.output_writers[:fields].filepath))
"""

using JLD2
using Plots
ENV["GKSwstype"] = "100"

function nice_divergent_levels(c, clim; nlevels=20)
    levels = range(-clim, stop=clim, length=nlevels)
    cmax = maximum(abs, c)
    clim < cmax && (levels = vcat([-cmax], levels, [cmax]))
    return (-clim, clim), levels
end

function nan_solid(y, z, v, seamount)
    Ny, Nz = size(v)
    y2 = reshape(y, Ny, 1)
    z2 = reshape(z, 1, Nz)
    v[seamount.(0, y2,z2)] .= NaN
    return nothing
end

function visualize_flow_over_seamount_simulation(prefix)

    h0, L = 0.5, 0.25                                        
    seamount(x, y, z) = z < - 1 + h0*exp(-y^2/L^2)
    
    filename = prefix * ".jld2"
    file = jldopen(filename)

    grid = file["serialized/grid"]


    xθ, yθ, zθ = nodes((Center, Center, Center), grid)

    θ₀ = file["timeseries/θ/0"][1, :, :]

    iterations = parse.(Int, keys(file["timeseries/t"]))    

    anim = @animate for (i, iter) in enumerate(iterations)

        @info "Plotting iteration $iter of $(iterations[end])..."

       
        θ = file["timeseries/θ/$iter"][1, :, :]
        t = file["timeseries/t/$iter"]

        θ′ = θ .- θ₀
	
	
        θmax = maximum(abs, θ)

        print("Max θ = ", θmax)

        
        nan_solid(yθ, zθ, θ′, seamount) 

        θ_title = @sprintf("θ, t = %.2f", t)

   
    θ_plot = contourf(yθ, zθ, θ'; title = θ_title)
	
    end

    mp4(anim, "flow_over_seamount.mp4", fps = 16)

    close(file)
end

visualize_flow_over_seamount_simulation("flow_over_seamount")
print("Simulation time = ", prettytime((finish_time - start_time)/1e9), "\n")

### here we compute the error for total tracer conservation
tracer_initial_tot
tracer_final_tot
analytical_error=(abs(tracer_initial_tot-tracer_final_tot)/tracer_initial_tot)*100

### here we compute the divergence of velocity
D = Field(Center, Center, Center, CPU(), grid_with_seamount)
D .= ∂y(V) + ∂z(W)
interior(D)[1,:,:]
