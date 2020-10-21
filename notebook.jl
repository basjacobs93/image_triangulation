### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ d36d547e-0fcc-11eb-1213-af7a3f5dab8a
using Images, SparseArrays

# ╔═╡ e15aff74-1386-11eb-31b2-a364ef6b249a
md"# Image Triangulation  
In this notebook, we will perform image triangulation based on the paper 'Stylized Image Triangulation'. The original code for the paper can be found [here](https://github.com/tobguent/image-triangulation). Since that is a combination of MatLab and C++, both of which I am not very familiar with, I found it difficult to follow. I therefore chose to implement this in Julia, which I want to learn, and which makes coding this very simple and fast.
"

# ╔═╡ 60345e14-0fcf-11eb-3252-65308a84851b
function generate_regular_grid(imwidth, imheight, n_points_x, n_points_y)
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
	
	
	# Build lists to return that make indexing faster
	neighbors = Int[] # list of all neighbors for all points
	neighbor_start_index = zeros(Int, 1, n_points) # index of first neighbor for point
	n_neighbors = zeros(Int, 1, n_points) # number of neighbors for point

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

	points, triangles, neighbors, neighbor_start_index, n_neighbors
end

# ╔═╡ aaa962be-1191-11eb-2a5a-910d2525765a
function xrange_yrange(triangle, width, height)	
	xs = min.(max.([t[1] for t in triangle], 1), width)
	ys = min.(max.([t[2] for t in triangle], 1), height)
	
	minimum(xs), maximum(xs), minimum(ys), maximum(ys)
end

# ╔═╡ a9a6cc34-1197-11eb-1bcf-198f3f9843cd
function sign(p1, p2, p3)
	(p1[1] - p3[1]) * (p2[2] - p3[2]) - (p2[1] - p3[1]) * (p1[2] - p3[2])
end

# ╔═╡ 8e67dbfc-1197-11eb-1ead-e7aee9325796
function isinside(triangle, point)
    d1 = sign(point, triangle[1], triangle[2])
    d2 = sign(point, triangle[2], triangle[3])
    d3 = sign(point, triangle[3], triangle[1])

    has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0)
    has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0)

    !(has_neg && has_pos)
end

# ╔═╡ e4fa3130-0fcc-11eb-0db2-af157823d58f
im = load("ekster.jpeg")

# ╔═╡ 11463f0a-1148-11eb-2e90-f9435eb460d1
im_rgb = channelview(im);

# ╔═╡ 43b92be8-1146-11eb-039b-8565ead70dfa
function triangle_color(triangle)
	# The triangle error is the difference of the triangle color (average) and
	# the points that lie within it
	r = Float32[]
	g = Float32[]
	b = Float32[]

	_, width, height = size(im_rgb)
	xmin, xmax, ymin, ymax = xrange_yrange(triangle, width, height)
	
	for x=xmin:xmax, y=ymin:ymax
		if isinside(triangle, [x, y])
			color = float(im_rgb[:, x, y])
			append!(r, color[1])
			append!(g, color[2])
			append!(b, color[3])
		end
	end
	trianglecolor = [sum(r)/length(r), sum(g)/length(g), sum(b)/length(b)]
		
	trianglecolor
end

# ╔═╡ 4733f55a-1196-11eb-08b4-3378b081ebc7
function triangle_error(triangle)
	# The triangle error is the difference of the triangle color (average) and
	# the points that lie within it
	r = Float32[]
	g = Float32[]
	b = Float32[]

	_, width, height = size(im_rgb)
	xmin, xmax, ymin, ymax = xrange_yrange(triangle, width, height)
	
	for x=xmin:xmax, y=ymin:ymax
		if isinside(triangle, [x, y])
			color = float(im_rgb[:, x, y])
			append!(r, color[1])
			append!(g, color[2])
			append!(b, color[3])
		end
	end
	trianglecolor = [sum(r)/length(r), sum(g)/length(g), sum(b)/length(b)]
	
	error = (sum((r .- trianglecolor[1]).^2) + sum((g .- trianglecolor[2]).^2) + sum((b .- trianglecolor[3]).^2)) / 3
	
	error
end

# ╔═╡ 9795130e-11ea-11eb-0e93-5fd4c59a31ad
function triangle_gradient(triangle, point_index)
	x1 = copy(triangle)
	x2 = copy(triangle)
	y1 = copy(triangle)
	y2 = copy(triangle)
	
	x1[point_index] += [1, 0]
	x2[point_index] -= [1, 0]
	
	y1[point_index] += [0, 1]
	y2[point_index] -= [0, 1]
	
	dx = (triangle_error(x1) - triangle_error(x2)) / 2
	dy = (triangle_error(y1) - triangle_error(y2)) / 2
	
	[dx, dy]
end

# ╔═╡ 432e03ec-0fcf-11eb-0335-05802e9ce929
begin
	width, height = size(im)
	n_points_x, n_points_y = 5, 5
end

# ╔═╡ 15e9619c-1194-11eb-3fda-a32a832f3b14
begin
	step_size = 1
	λ = 1e-3
	
	# Generate initial grid
	points, triangles, neighbors, neighbor_start_index, n_neighbors = generate_regular_grid(width, height, n_points_x, n_points_y)
	
	n_triangles = size(triangles)[2]
	n_points = size(points)[2]
end

# ╔═╡ af3ad1ec-12b2-11eb-0b5f-e5732664b072
function find_1ring(point_index, neighbors, neighbor_start_index, n_neighbors, triangles)
	# Return all vertices that are on triangles adjacent to a certain point,
	# excluding the point itself
	first_neighbor = neighbor_start_index[point_index]
	last_neighbor = first_neighbor + n_neighbors[point_index] - 1
	
	# Adjacent triangle indices
	ns = neighbors[first_neighbor:last_neighbor]
	
	# 1ring-point indices
	pi = [p for n in ns for p in triangles[:, n] if p != point_index]
	
	[points[:, i] for i in unique(pi)]
end

# ╔═╡ 83a210bc-12b3-11eb-0355-d339e466a369
function regularization(point_index, neighbors, neighbor_start_index, n_neighbors, triangles)
	# Move a point towards the center of the points of the adjacent triangles
	
	point = points[:, point_index]
	
	ns = find_1ring(point_index, neighbors, neighbor_start_index, n_neighbors, triangles)
		
	(sum(n for n in ns) ./ length(ns)) - point
end

# ╔═╡ 23c20d76-117f-11eb-297e-0376a11533bd
function point_gradient(point_index, neighbors, neighbor_start_index, n_neighbors, triangles)
	first_neighbor = neighbor_start_index[point_index]
	last_neighbor = first_neighbor + n_neighbors[point_index] - 1
	
	ns = neighbors[first_neighbor:last_neighbor]
	
	grad = [0., 0.]
	for i in ns
		# triangle = triangle_objects[i]
		triangle_points = triangles[:, i]
		which_point = findfirst(triangle_points .== point_index)
		
		triangle = [points[:, j] for j in triangle_points]
		grad = grad .+ triangle_gradient(triangle, which_point)
	end
	
	reg = regularization(point_index, neighbors, neighbor_start_index, n_neighbors, triangles)
		
	grad
end

# ╔═╡ e761f3b8-117f-11eb-12b7-6f1c2a3d0977
function draw_image(triangles, colors)
	img = zeros(RGB{Float32}, width, height)
	
	for x=1:width, y=1:height
		for i in 1:size(triangles)[2]
			if isinside([points[:, j] for j in triangles[:, i]], [x; y])
				img[x, y] = RGB(colors[i]...)
				break
			end
		end
	end
	
	img
end

# ╔═╡ 7489e73c-1211-11eb-2242-9f477e72fc2a
function update_point(point, step_size, grad, reg, width, height)
	new_point = round.(Int, point - (step_size * grad) + (λ * reg))
	
	# Make sure point is within bounds
	new_point[1] = max(min(new_point[1], width), 1)
	new_point[2] = max(min(new_point[2], height), 1)

	# Do not move boundary points
	if point[1] == 1
		new_point[1] = 1
	elseif point[1] == height
		new_point[1] = height
	end
	if point[2] == 1
		new_point[2] = 1
	elseif point[2] == width
		new_point[2] = width
	end
	
	new_point
end

# ╔═╡ 562a1e58-118b-11eb-28a1-c576d975bb6d
for j in 1:40
	triangle_objects = [[points[:, i] for i in triangles[:, j]] for j in 1:n_triangles]
	triangle_colors = [triangle_color(triangle) for triangle in triangle_objects]
	
	# Update points
	for i in 1:n_points
		grad = point_gradient(i, neighbors, neighbor_start_index, n_neighbors, triangles)
		reg = regularization(i, neighbors, neighbor_start_index, n_neighbors, triangles)

		points[:, i] = update_point(points[:, i], step_size, grad, reg, width, height)
	end
end

# ╔═╡ ad415226-1195-11eb-2b9f-73142b04de89
begin
	triangle_objects = [[points[:, i] for i in triangles[:, j]] for j in 1:n_triangles]

	triangle_colors = [triangle_color(triangle) for triangle in triangle_objects]

	draw_image(triangles, triangle_colors)
end

# ╔═╡ Cell order:
# ╠═e15aff74-1386-11eb-31b2-a364ef6b249a
# ╠═d36d547e-0fcc-11eb-1213-af7a3f5dab8a
# ╠═60345e14-0fcf-11eb-3252-65308a84851b
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
# ╠═e4fa3130-0fcc-11eb-0db2-af157823d58f
# ╠═11463f0a-1148-11eb-2e90-f9435eb460d1
# ╠═432e03ec-0fcf-11eb-0335-05802e9ce929
# ╠═15e9619c-1194-11eb-3fda-a32a832f3b14
# ╠═562a1e58-118b-11eb-28a1-c576d975bb6d
# ╠═ad415226-1195-11eb-2b9f-73142b04de89
