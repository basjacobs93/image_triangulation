### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ d36d547e-0fcc-11eb-1213-af7a3f5dab8a
using Images, SparseArrays, VoronoiDelaunay, StatsBase

# ╔═╡ e15aff74-1386-11eb-31b2-a364ef6b249a
md"# Image Triangulation  
In this notebook, we will perform image triangulation based on the paper 'Stylized Image Triangulation' by Kai Lawonn and Tobias Günther. The paper can be found [here](https://cgl.ethz.ch/Downloads/Publications/Papers/2018/Law18a/Law18a.pdf) and the original code can be found [here](https://github.com/tobguent/image-triangulation). Since that is a combination of MatLab and C++ code, neither of which I am very familiar with, I found it difficult to follow. I therefore chose to implement this in Julia, which I want to learn, and which makes coding this very simple and fast.
"

# ╔═╡ 60345e14-0fcf-11eb-3252-65308a84851b
function generate_regular_grid(imwidth::Int64, imheight::Int64,
							   n_points_x::Int64, n_points_y::Int64)::Tuple{Array{Int64,2}, Array{Int64,2}}
	# Create points: a 2 x n_points array containing point coordinates
	# Port from https://github.com/tobguent/image-triangulation
	n_points = n_points_x * n_points_y
	
	points = zeros(Int, 2, n_points)
	
	tmp = round.(Int, (0:n_points_x-1) .* ((imwidth-1)/(n_points_x-1)).+1)
	points[1, :] = repeat(tmp, 1, n_points_y)

	tmp = round.(Int, (0:n_points_y-1) .* ((imheight-1)/(n_points_y-1)).+1)
	tmp = repeat(tmp, 1, n_points_x)
	points[2, :] = reshape(tmp', n_points)
	
	
	# Create triangles: a 3 x n_triangles array containing point indices
	n_triangles = 2 * (n_points_x-1) * (n_points_y-1)
	triangles = zeros(Int, 3, n_triangles)
	
	tmp = zeros(1, 2 * (n_points_x - 1))
	tmp[1:2:end-1] = 1:n_points_x-1
	tmp[2:2:end] = 2:n_points_x

	tmp2 = reshape(collect(1:n_triangles), 1, n_triangles)
	tmp2 = (tmp2 - mod.(tmp2.-1, 2*(n_points_x-1)) .- 1) / (2 * (n_points_x-1))
	triangles[1, :] = repeat(tmp, 1, n_points_y-1) + (tmp2 .* n_points_x)

	triangles[3, 1:2:end] = triangles[1, 1:2:end] .+ n_points_x
	triangles[3, 2:2:end] = triangles[1, 2:2:end] .+ n_points_x.-1

	triangles[2, 1:2:end] = triangles[1, 1:2:end] .+ 1
	triangles[2, 2:2:end] = triangles[3, 2:2:end] .+ 1
	
	
	points, triangles
end

# ╔═╡ 9c6d7056-1474-11eb-0a3c-a1568b4f5c14
function convolve(im::Array{Float32,2}, kernel::Array{Int64,2})::Array{Float64,2}
	width, height = size(im)
	kernel_width, kernel_height = size(kernel)
	
	im_new = copy(im)
	
	margin = (kernel_width-1)÷2
	
	for x in (margin+1):(width-margin)
		for y in (margin+1):(height-margin)
			im_new[y, x] = sum(im[(y-margin):(y+margin), (x-margin):(x+margin)] * kernel)
		end
	end
	
	im_new
end

# ╔═╡ e379faf0-1476-11eb-0a1f-db6d6ef2c58f
function generate_importance_grid(im_gray::Array{Float32,2}, width::Int64,
								  height::Int64, n_points::Int64)::Tuple{Array{Int64,2}, Array{Int64,2}}
	# Generate a grid based on importance of points via a simple edge detection,
	# connecting these points with a delaunay 
	kernel = [-1 -1 -1; -1 8 -1; -1 -1 -1]
	edges = convolve(im_gray, kernel)
	
	coords = [(x, y) for x in 1:width for y in 1:height]
	weights = [edges[x, y] for x in 1:width for y in 1:height]
	weights = weights .- minimum(weights) # make sure weights are positive
	
	# Sample points
	points = sample(coords, Weights(weights), n_points-4, replace=false)
	# Add 4 corners
	append!(points, [(1, 1), (1, height), (width, 1), (width, height)])

	# Scale to [1, 2] since that's needed by DelaunayTesselation
	offset = 1.01
	scale = 0.98 / max( height - 1, width - 1 )
	
	# Create tesselation
	point_list = Point2D[Point(( point[1]-1 ) * scale + offset, (point[2]-1) * scale + offset) for point in points]
	tess = DelaunayTessellation2D(n_points)
	push!(tess, point_list)
	
	# Convert to useful form for our algorithm
	n_triangles = tess._last_trig_index
	n_points = length(point_list)
	
	triangles = zeros(Int, 3, n_triangles)
	points = zeros(Int, 2, n_points)

	# Rescale points to original range
	for point_index in 1:n_points
		pt_scaled = point_list[point_index]
		pt = round.(Int, [(getx(pt_scaled) - offset) / scale + 1,
						  (gety(pt_scaled) - offset) / scale + 1])
		points[:, point_index] = pt
	end

	# Create triangles array
	for triangle_index in 1:n_triangles
		tri = tess._trigs[triangle_index]
		pts = [geta(tri), getb(tri), getc(tri)]
		pts = [round.(Int, [min(max((getx(pt) - offset) / scale + 1, 1), width),
						  min(max((gety(pt) - offset) / scale + 1, 1), height)])
			   for pt in pts]


		for i in 1:3
			pi = findfirst([points[:, j] == pts[i] for j in 1:n_points])
			
			triangles[i, triangle_index] = pi
		end
	end
	
	points, triangles[:, 2:end]
end

# ╔═╡ 0da11472-148b-11eb-3a01-6f4fff588360
function generate_neighbors_lists(points::Array{Int64,2}, triangles::Array{Int64,2})::Tuple{Array{Int64,1},Array{Int64,1},Array{Int64,1}}
	# Build lists to return that make indexing faster
	n_points = size(points)[2]
	
	neighbors = Int[] # list of all neighbors for all points
	neighbor_start_index = zeros(Int, n_points) # index of first neighbor for point
	n_neighbors = zeros(Int, n_points) # number of neighbors for point

	for i = 1:n_points
		# triangles adjacent to point
		tri_n = [ind[2] for ind in findall(triangles .== i)] 
		append!(neighbors, tri_n)
		n_neighbors[i] = length(tri_n)
		if i == 1
			neighbor_start_index[i] = 1;
		else
			neighbor_start_index[i] = neighbor_start_index[i-1] + n_neighbors[i-1];
		end
	end
	
	neighbors, neighbor_start_index, n_neighbors
end

# ╔═╡ aaa962be-1191-11eb-2a5a-910d2525765a
function xrange_yrange(triangle::Array{Int64,2}, width::Int64,
					   height::Int64)::Tuple{Int64,Int64,Int64,Int64}
	# Return the smallest and largest coordinates that could be inside
	# the triangle
	xs = min.(max.(triangle[1,:], 1), width)
	ys = min.(max.(triangle[2,:], 1), height)
	
	minimum(xs), maximum(xs), minimum(ys), maximum(ys)
end

# ╔═╡ af3ad1ec-12b2-11eb-0b5f-e5732664b072
function find_1ring(point_index::Int64, points::Array{Int64,2},
					triangles::Array{Int64,2}, adjacent_triangles::Array{Int64,1}
				   )::Array{Array{Int64,1},1}
	# Return all vertices that are on triangles adjacent to a certain point,
	# excluding the point itself
	# 1ring-point indices
	pis = [pi for ti in adjacent_triangles for pi in triangles[:, ti] if pi != point_index]
	
	[points[:, i] for i in unique(pis)]
end

# ╔═╡ 83a210bc-12b3-11eb-0355-d339e466a369
function regularization(point_index::Int64, points::Array{Int64,2}, 
						triangles::Array{Int64,2},
						adjacent_triangles::Array{Int64,1})::Array{Float64,1}
	# Move a point towards the center of the points of the adjacent triangles
	
	point = points[:, point_index]
	
	ns = find_1ring(point_index, points, triangles, adjacent_triangles)
		
	(sum(n for n in ns) ./ length(ns)) - point
end

# ╔═╡ 7489e73c-1211-11eb-2242-9f477e72fc2a
function update_point(point::Array{Int64,1}, step_size::Float64, λ::Float64,
					  grad::Array{Float64,1}, reg::Array{Float64,1},
					  width::Int64, height::Int64)::Array{Int64,1}
	new_point = round.(Int, point - (step_size * grad) + (λ * reg))
	
	# Make sure point is within bounds
	new_point[1] = max(min(new_point[1], width), 1)
	new_point[2] = max(min(new_point[2], height), 1)

	# Do not move boundary points
	if point[1] == 1
		new_point[1] = 1
	elseif point[1] == width
		new_point[1] = width
	end
	if point[2] == 1
		new_point[2] = 1
	elseif point[2] == height
		new_point[2] = height
	end
	
	new_point
end

# ╔═╡ a9a6cc34-1197-11eb-1bcf-198f3f9843cd
function sign(p1::Array{Int64,1}, p2::Array{Int64,1}, p3::Array{Int64,1})::Int64
	(p1[1] - p3[1]) * (p2[2] - p3[2]) - (p2[1] - p3[1]) * (p1[2] - p3[2])
end

# ╔═╡ 8e67dbfc-1197-11eb-1ead-e7aee9325796
function isinside(triangle::Array{Int64,2}, point::Array{Int64,1})::Bool
    d1 = sign(point, triangle[:,1], triangle[:,2])
    d2 = sign(point, triangle[:,2], triangle[:,3])
    d3 = sign(point, triangle[:,3], triangle[:,1])

    has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0)
    has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0)

    !(has_neg && has_pos)
end

# ╔═╡ 43b92be8-1146-11eb-039b-8565ead70dfa
function triangle_color(triangle::Array{Int64,2}, im_rgb::Array{N0f8,3})::Array{Float64,1}
	# The triangle color is the average of the color of
	# points within the triangle
	r = Float32[]
	g = Float32[]
	b = Float32[]

	_, height, width = size(im_rgb)
	xmin, xmax, ymin, ymax = xrange_yrange(triangle, width, height)
	
	for x=xmin:xmax, y=ymin:ymax
		if isinside(triangle, [x, y])
			color = im_rgb[:, y, x]
			append!(r, color[1])
			append!(g, color[2])
			append!(b, color[3])
		end
	end
	trianglecolor = [sum(r)/length(r), sum(g)/length(g), sum(b)/length(b)]
		
	trianglecolor
end

# ╔═╡ 4733f55a-1196-11eb-08b4-3378b081ebc7
function triangle_error(triangle::Array{Int64,2}, im_rgb::Array{N0f8,3}; mean = false)::Float64
	# The triangle error is the difference of the triangle color (average) and
	# the points that lie within it
	r = Float32[]
	g = Float32[]
	b = Float32[]

	n_points = 0 # Number of points inside triangle
	_, height, width = size(im_rgb)
	xmin, xmax, ymin, ymax = xrange_yrange(triangle, width, height)
	
	for x=xmin:xmax, y=ymin:ymax
		if isinside(triangle, [x, y])
			n_points = n_points + 1
			color = im_rgb[:, y, x]
			append!(r, color[1])
			append!(g, color[2])
			append!(b, color[3])
		end
	end
	trianglecolor = [sum(r)/length(r), sum(g)/length(g), sum(b)/length(b)]
	
	error = (sum((r .- trianglecolor[1]).^2) + sum((g .- trianglecolor[2]).^2) + sum((b .- trianglecolor[3]).^2)) / 3
	
	if mean
		error/n_points
	else
		error
	end
end

# ╔═╡ 9795130e-11ea-11eb-0e93-5fd4c59a31ad
function triangle_gradient(triangle::Array{Int64,2}, point_index::Int64,
						   im_rgb::Array{N0f8,3})::Array{Float64,1}
	# Calculate the gradient of triangle with respect to point_index
	x1 = copy(triangle)
	x2 = copy(triangle)
	y1 = copy(triangle)
	y2 = copy(triangle)
	
	x1[:, point_index] += [1, 0]
	x2[:, point_index] -= [1, 0]
	
	y1[:, point_index] += [0, 1]
	y2[:, point_index] -= [0, 1]
	
	dx = (triangle_error(x1, im_rgb) - triangle_error(x2, im_rgb)) / 2
	dy = (triangle_error(y1, im_rgb) - triangle_error(y2, im_rgb)) / 2
	
	[dx, dy]
end

# ╔═╡ 23c20d76-117f-11eb-297e-0376a11533bd
function point_gradient(point_index::Int64, points::Array{Int64,2},
					    triangles::Array{Int64,2},
						adjacent_triangles::Array{Int64,1}, im_rgb::Array{N0f8,3})::Array{Float64,1}

	grad = [0., 0.]
	for i in adjacent_triangles
		triangle_points = triangles[:, i]
		which_point = findfirst(triangle_points .== point_index)
		
		triangle = points[:, triangle_points]
		grad = grad .+ triangle_gradient(triangle, which_point, im_rgb)
	end
	
	grad
end

# ╔═╡ e761f3b8-117f-11eb-12b7-6f1c2a3d0977
function draw_image(triangles::Array{Int64,2}, points::Array{Int64,2}, 
					im_rgb::Array{N0f8,3}, width::Int64, height::Int64)::Array{ColorTypes.RGB{Float32},2}
	img = zeros(RGB{Float32}, height, width)
	
	triangle_colors = []
	for i in 1:size(triangles)[2]
		triangle = points[:, triangles[:, i]]
		push!(triangle_colors, triangle_color(triangle, im_rgb))
	end
	
	for x=1:width, y=1:height
		for i in 1:size(triangles)[2]
			triangle = points[:, triangles[:, i]]
			if isinside(triangle, [x; y])
				color = triangle_colors[i]
				img[y, x] = RGB(color...)
				break
			end
		end
	end
	
	img
end

# ╔═╡ 99ead454-1fa0-11eb-0c97-b159b530ceaa
function centroid(triangle::Array{Int64,2})::Array{Int64,1}
	[round.(Int, mean(triangle[1, :])), round.(Int, mean(triangle[2, :]))]
end

# ╔═╡ 9ee47e42-2522-11eb-3253-d5a314d33b44
function triangle_size(triangle)::Float64
	width = maximum(triangle[1, :]) - minimum(triangle[1, :])
	height = maximum(triangle[2, :]) - minimum(triangle[2, :])
	min(width, height)
end

# ╔═╡ c86c8d8c-2433-11eb-3c26-e51f78d97a9c
function split_triangle(triangles::Array{Int64,2}, points::Array{Int64,2}, i::Int64)::Tuple{Array{Int64,2}, Array{Int64,2}}
	triangle = triangles[:, i]
	
	new_point = centroid(points[:, triangle])
	
	new_points = hcat(points, new_point)
	new_point_index = size(new_points)[2]
	
	triangles[:,i] = [triangle[1], triangle[2], new_point_index]
	new_triangles = hcat(hcat(triangles,
							  [triangle[1], new_point_index, triangle[3]]),
		 				 [new_point_index, triangle[2], triangle[3]])
	
	new_points, new_triangles
end

# ╔═╡ 562a1e58-118b-11eb-28a1-c576d975bb6d
function gradient_descent(points::Array{Int64,2}, triangles::Array{Int64,2},
						  neighbors::Array{Int64,1}, neighbor_start_index::
Array{Int64,1},
						  n_neighbors::Array{Int64,1}, im_rgb::Array{N0f8,3},
						  width::Int64, height::Int64, n_steps::Int64,
						  step_size::Float64, λ::Float64, τ::Float64, 
						  max_n_triangles::Int64,
						  min_triangle_size::Float64
						  )::Tuple{Array{Int64,2}, Array{Int64,2}}
	
	for j in 1:n_steps
		# Number of points and triangles can increase each step
		n_triangles = size(triangles)[2]
		n_points = size(points)[2]

		# Move points towards their gradient
		for i in 1:n_points
			point = points[:, i]

			# Adjacent triangle indices
			first_neighbor = neighbor_start_index[i]
			last_neighbor = first_neighbor + n_neighbors[i] - 1
			adjacent_triangles = neighbors[first_neighbor:last_neighbor]

			grad = point_gradient(i, points, triangles, adjacent_triangles, im_rgb)
			reg = regularization(i, points, triangles, adjacent_triangles)

			points[:, i] = update_point(point, step_size, λ, grad, reg, width, height)
		end
		
		# Add triangles if error larger than threshold
		if n_triangles < max_n_triangles
			for i in 1:n_triangles
				triangle = points[:, triangles[:, i]]
				rmtae = sqrt(triangle_error(triangle, im_rgb; mean = true))
				size = triangle_size(triangle)
				if (rmtae > τ) & (size > min_triangle_size)
					points, triangles = split_triangle(triangles, points, i)
					neighbors, neighbor_start_index, n_neighbors = generate_neighbors_lists(points, triangles)
				end
			end
		end
	end
	
	points, triangles
end

# ╔═╡ 21c14a46-13c8-11eb-2780-2f4d79801128
im = load("baardman.jpg")

# ╔═╡ ad415226-1195-11eb-2b9f-73142b04de89
begin	
	height, width = size(im)
	im_rgb = real.(channelview(im))
	im_gray = gray.(float(Gray.(im)))
	

	step_size = 2.
	n_steps = 200
	λ = 0.05 # regularization size
	τ = 0.2 # threshold to split triangle
	min_triangle_size = 1/5 * min(height, width) # only split triangle if not too small
	max_n_triangles = 100 # stop splitting if reached
	
	# Generate initial grid
	# points, triangles = generate_importance_grid(im_gray, width, height, 5)
	points, triangles = generate_regular_grid(width, height, 5, 5)
 	neighbors, neighbor_start_index, n_neighbors = generate_neighbors_lists(points, triangles)
	
	# Perform gradient descent
	points, triangles = gradient_descent(points, triangles, neighbors,
										 neighbor_start_index, n_neighbors,
										 im_rgb, width, height,
										 n_steps, step_size, λ, τ,
										 max_n_triangles, min_triangle_size)
	
	# Draw output image
	draw_image(triangles, points, im_rgb, width, height)
end

# ╔═╡ 7ed0a3c0-2526-11eb-1749-bfd5d1d2eb6f
[triangle_size(points[:, triangles[:, i]]) for i in 1:size(triangles)[2]]

# ╔═╡ Cell order:
# ╠═e15aff74-1386-11eb-31b2-a364ef6b249a
# ╠═d36d547e-0fcc-11eb-1213-af7a3f5dab8a
# ╠═60345e14-0fcf-11eb-3252-65308a84851b
# ╠═e379faf0-1476-11eb-0a1f-db6d6ef2c58f
# ╠═9c6d7056-1474-11eb-0a3c-a1568b4f5c14
# ╠═0da11472-148b-11eb-3a01-6f4fff588360
# ╠═aaa962be-1191-11eb-2a5a-910d2525765a
# ╠═43b92be8-1146-11eb-039b-8565ead70dfa
# ╠═4733f55a-1196-11eb-08b4-3378b081ebc7
# ╠═9795130e-11ea-11eb-0e93-5fd4c59a31ad
# ╠═23c20d76-117f-11eb-297e-0376a11533bd
# ╠═af3ad1ec-12b2-11eb-0b5f-e5732664b072
# ╠═83a210bc-12b3-11eb-0355-d339e466a369
# ╠═e761f3b8-117f-11eb-12b7-6f1c2a3d0977
# ╠═7489e73c-1211-11eb-2242-9f477e72fc2a
# ╠═a9a6cc34-1197-11eb-1bcf-198f3f9843cd
# ╠═8e67dbfc-1197-11eb-1ead-e7aee9325796
# ╠═99ead454-1fa0-11eb-0c97-b159b530ceaa
# ╠═9ee47e42-2522-11eb-3253-d5a314d33b44
# ╠═562a1e58-118b-11eb-28a1-c576d975bb6d
# ╠═c86c8d8c-2433-11eb-3c26-e51f78d97a9c
# ╠═21c14a46-13c8-11eb-2780-2f4d79801128
# ╠═ad415226-1195-11eb-2b9f-73142b04de89
# ╠═7ed0a3c0-2526-11eb-1749-bfd5d1d2eb6f
