ENV["GKSwstype"] = "nul"
using Printf
using Oceananigans
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Plots
using JLD2
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
V_difference = YFaceField(CPU(), grid_with_seamount)
### W is (Center, Centere, Face)
W_difference = ZFaceField(CPU(), grid_with_seamount)

### V and W are deravatives of psi 
### which in this code we use deravative functions
### (which are defined in julia) to find the values
V_difference.=  ∂z(Ψ)
W_difference.= -∂y(Ψ)

### We fill the halo regions of V and W
fill_halo_regions!(V_difference, CPU())
fill_halo_regions!(W_difference, CPU())

### We mask V and W        
mask_immersed_field!(V_difference)
mask_immersed_field!(W_difference)

### V is (Center, Face, Center)
V_analyc = YFaceField(CPU(), grid_with_seamount)
### W is (Center, Centere, Face)
W_analyc= ZFaceField(CPU(), grid_with_seamount)

### V and W are deravatives of psi 
### which in this code we use exact expression 
### which we computed in maple
### we are calling this method 'analytical'
set!(V_analyc, (x, y, z) -> -2 * (1 - ( z / (-1+h0*exp(-y^2/L^2)) ) )/(-1+h0*exp(-y^2/L^2)))
set!(W_analyc, (x, y, z) -> 4*h0*z*y*exp(-y^2/L^2)*(1 - z/(-1 + h0*exp(-y^2/L^2)))/((L^2)*(-1 + h0*exp(-y^2/L^2))^2))

### We fill the halo regions of V and W
fill_halo_regions!(V_analyc, CPU())
fill_halo_regions!(W_analyc, CPU())

### We mask V and W        
mask_immersed_field!(V_analyc)
mask_immersed_field!(W_analyc)


xv, yv, zv = nodes((Center, Face, Center), grid)

        v_analyc_plot = contourf(yv, zv, interior(V_analyc)[1,:,:]'; title = "V_analytical")
        w_analyc_plot = contourf(yv, zv, interior(W_analyc)[1,:,:]'; title = "W_analytical")
        v_deravtive_plot = contourf(yv, zv, interior(V_difference)[1,:,:]'; title = "V_difference")
        w_deravtive_plot = contourf(yv, zv, interior(W_difference)[1,:,:]'; title = "W_difference")

        #plt_v_final=plot(v_analyc_plot, w_analyc_plot, v_deravtive_plot ,w_deravtive_plot, layout = (2, 2), size = (1200, 1200))
        plt_v_final=plot(v_analyc_plot, v_deravtive_plot, layout = (2, 1), size = (1200, 1200))
savefig(plt_v_final, "comparing_velocities.png")
maximum(interior(V_analyc)[1,:,:])
argmax(interior(V_analyc)[1,:,:])
maximum(interior(V_difference)[1,:,:])
argmax(interior(V_difference)[1,:,:])
interior(V_analyc)[1,:,:]./interior(V_difference)[1,:,:]
