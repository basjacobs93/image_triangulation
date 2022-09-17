### A Pluto.jl notebook ###
# v0.19.8

using Markdown
using InteractiveUtils

# ╔═╡ d36d547e-0fcc-11eb-1213-af7a3f5dab8a
begin
	using Images, SparseArrays, VoronoiDelaunay, StatsBase
	import Luxor
end

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
	# Apply convolution on `im` given `kernel` 
	height, width = size(im)
	kernel_height, kernel_width = size(kernel)
	
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
								  height::Int64, n_points::Int64)::Tuple{Array{Int64,2},Array{Int64,2}}
	# Generate a grid based on importance of points via a simple edge detection,
	# connecting these points with a delaunay 
	kernel = [-1 -1 -1; -1 8 -1; -1 -1 -1]
	edges = convolve(im_gray, kernel)
	
	coords = [(x, y) for x in 1:width for y in 1:height]
	weights = [edges[y, x] for x in 1:width for y in 1:height]
	weights = weights .- minimum(weights) # make sure weights are positive
	
	# Sample points
	points = sample(coords, Weights(weights), n_points-4, replace=false)
	# Add 4 corners
	append!(points, [(1, 1), (1, height), (width, 1), (width, height)])
	# Add 4 points in centers of edges
	append!(points, [(floor(width/2), 1), (1, floor(height/2)), (floor(width/2), height), (width, floor(height/2))])

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

	triangles = triangles[:, 2:end]

	# Remove triangles for which two points are identical
	keep_triangles = [i for i in 1:(n_triangles-1) if length(unique(triangles[:, i])) == 3]
	triangles = triangles[:, keep_triangles]
	
	points, triangles
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
	# Update point location, satisfying width and height of image
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
	# Helper function for `isinside`
	(p1[1] - p3[1]) * (p2[2] - p3[2]) - (p2[1] - p3[1]) * (p1[2] - p3[2])
end

# ╔═╡ 8e67dbfc-1197-11eb-1ead-e7aee9325796
function isinside(triangle::Array{Int64,2}, point::Array{Int64,1})::Bool
	# Returns boolean whether point lies inside triangle
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

# ╔═╡ e761f3b8-117f-11eb-12b7-6f1c2a3d0977
function draw_image(triangles::Array{Int64,2}, points::Array{Int64,2}, 
					im_rgb::Array{N0f8,3}, width::Int64, height::Int64, 
					scale::Int64, i::Int64)::Luxor.Drawing	
	Luxor.@svg begin
	
		Luxor.origin(0, 0)
		Luxor.background("white")
		
		for i in 1:size(triangles)[2]
			triangle = points[:, triangles[:, i]]
			color = RGB(Float32.(triangle_color(triangle, im_rgb))...)
			Luxor.sethue(color)
			verts = [Luxor.Point(scale.*triangle[:,1]...),
					 Luxor.Point(scale.*triangle[:,2]...),
					 Luxor.Point(scale.*triangle[:,3]...)]
	
			Luxor.poly(verts, :fill)
			Luxor.poly(verts, :stroke)
		end
	end width*scale height*scale "tmp/" * string(i+1)
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
	# Calculate the gradient for the point at index `point_index`

	grad = [0., 0.]
	for i in adjacent_triangles
		triangle_points = triangles[:, i]
		which_point = findfirst(triangle_points .== point_index)
		
		triangle = points[:, triangle_points]
		grad = grad .+ triangle_gradient(triangle, which_point, im_rgb)
	end
	
	grad./length(adjacent_triangles)
end

# ╔═╡ 99ead454-1fa0-11eb-0c97-b159b530ceaa
function centroid(triangle::Array{Int64,2})::Array{Int64,1}
	# Returns center of triangle
	[round.(Int, mean(triangle[1, :])), round.(Int, mean(triangle[2, :]))]
end

# ╔═╡ 9ee47e42-2522-11eb-3253-d5a314d33b44
function triangle_size(triangle)::Float64
	# Returns the length of the longest side of the triangle
	x1, x2, x3 = triangle[1, :]
	y1, y2, y3 = triangle[2, :]
	
	s1 = sqrt((x1-x2)^2 + (y1-y2)^2)
	s2 = sqrt((x2-x3)^2 + (y2-y3)^2)
	s3 = sqrt((x3-x1)^2 + (y3-y1)^2)
	
	max(s1, s2, s3)
end

# ╔═╡ ea76e3f6-6ef6-4375-a635-b8720fab5602
function triangle_area(triangle)::Float64
	# Returns the area of a triangle
	x1, x2, x3 = triangle[1, :]
	y1, y2, y3 = triangle[2, :]
	
	s1 = sqrt((x1-x2)^2 + (y1-y2)^2)
	s2 = sqrt((x2-x3)^2 + (y2-y3)^2)
	s3 = sqrt((x3-x1)^2 + (y3-y1)^2)
	
	area = abs(x1*y2 + x2*y3 + x3*y1 - y1*x2 - y2*x3 - y3*x1)
	
	area
end

# ╔═╡ 66877a86-2985-11eb-3e14-439d0b057cbb
function triangle_stretchedness(triangle)::Float64
	# Returns the ratio between the longest side and the height of a triangle
	x1, x2, x3 = triangle[1, :]
	y1, y2, y3 = triangle[2, :]
	
	s1 = sqrt((x1-x2)^2 + (y1-y2)^2)
	s2 = sqrt((x2-x3)^2 + (y2-y3)^2)
	s3 = sqrt((x3-x1)^2 + (y3-y1)^2)
	
	width = max(s1, s2, s3)
	area = abs(x1*y2 + x2*y3 + x3*y1 - y1*x2 - y2*x3 - y3*x1)
	height = area/width
	
	width/height
end

# ╔═╡ 3381b436-485d-4ec6-b80c-aa9ddad6624d
function angle_from_points(point1::Array{Int64}, point2::Array{Int64}, point3::Array{Int64})::Float64
	# Find interior angle of point1 in triangle point1, point2, point3
	x1, y1 = point1
	x2, y2 = point2
	x3, y3 = point3

	s1 = sqrt((x2-x3)^2 + (y2-y3)^2)
	s2 = sqrt((x3-x1)^2 + (y3-y1)^2)
	s3 = sqrt((x1-x2)^2 + (y1-y2)^2)

	cos_num = (s2^2 + s3^2 - s1^2)/(2*s2*s3)

	acos(max(min(cos_num, 1.), -1.))
end

# ╔═╡ 3480aa2c-2e71-4d62-ad15-fc38b17184ba
function find_flippable_edges(points::Array{Int64,2}, triangles::Array{Int64,2})::Tuple{Vector{Int64}, Vector{Int64}, Tuple{Int64, Int64}} 
	# Find edges that can be flipped
	n_triangles = size(triangles)[2]
	
	for t in 1:n_triangles
		nb = Int[]
		for i in triangles[:, t]
			# Other triangles adjacent to point
			tri_n = [ind[2] for ind in findall(triangles .== i)] 
			append!(nb, tri_n)
		end
		common_triangles = [tri for (tri, cnt) in countmap(nb) if ((cnt > 1) & (tri != t))]
		if length(common_triangles) > 0
			for t2 in common_triangles
				common_points = intersect(triangles[:, t], triangles[:, t2])
				opposite_points = symdiff(triangles[:, t], triangles[:, t2])

				if length(common_points) > 2
					return [-3], [-3], (t, t2)
				end

				# println(opposite_points) leeg
				# println(common_points) 3 stuks
				angle1 = angle_from_points(points[:, opposite_points[1]], points[:, common_points[1]], points[:, common_points[2]])
				angle2 = angle_from_points(points[:, opposite_points[2]], points[:, common_points[1]], points[:, common_points[2]])

				if angle1+angle2 > pi + 0.0001 # tie breaker to prevent back and forths
					return common_points, opposite_points, (t, t2)
				end
			end
		end
	end
	
	return [-1], [-1], (-1, -1)
end

# ╔═╡ c86c8d8c-2433-11eb-3c26-e51f78d97a9c
function split_triangle(triangles::Array{Int64,2}, points::Array{Int64,2}, i::Int64)::Tuple{Array{Int64,2}, Array{Int64,2}}
	# Split triangle into three parts by adding a point at its centroid
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

# ╔═╡ fa734135-983f-4ea2-b3f3-fee90a97dc26
function remove_point(triangles::Array{Int64,2}, points::Array{Int64,2}, i::Int64, neighbors::Array{Int64,1},neighbor_start_index::Array{Int64,1}, n_neighbors::Array{Int64,1})::Tuple{Array{Int64,2}, Array{Int64,2}}
	# Remove point, thus merging three triangles
	
	# Adjacent triangle indices
	first_neighbor = neighbor_start_index[i]
	last_neighbor = first_neighbor + n_neighbors[i] - 1
	adjacent_triangles = neighbors[first_neighbor:last_neighbor]

	if length(adjacent_triangles) == 3
		triangle_points = []
		for triangle in adjacent_triangles
			append!(triangle_points, triangles[:, triangle])
		end
		triangle_points = unique([pt for pt in triangle_points if pt != i])
	
		# Collapse onto first triangle
		if length(triangle_points) != 3
			return points, triangles
		end
		triangles[:, adjacent_triangles[1]] = triangle_points

		n_triangles = size(triangles)[2]
		keep_triangles = [i for i in 1:n_triangles if i != adjacent_triangles[2] && i != adjacent_triangles[3]]
		triangles = triangles[:, keep_triangles]
	end
	
	points, triangles
end

# ╔═╡ 562a1e58-118b-11eb-28a1-c576d975bb6d
function gradient_descent(points::Array{Int64,2}, triangles::Array{Int64,2},
						  neighbors::Array{Int64,1},
						  neighbor_start_index::Array{Int64,1},
						  n_neighbors::Array{Int64,1}, im_rgb::Array{N0f8,3},
						  width::Int64, height::Int64, n_steps::Int64,
						  step_size::Float64, λ::Float64, τ::Float64, 
						  split_every::Int64,
						  max_n_triangles::Int64,
						  min_triangle_size::Float64,
						  max_triangle_stretchedness::Float64,
						  min_area::Float64
						  )::Tuple{Array{Int64,2}, Array{Int64,2}}
	# Perform `n_steps` of gradient descent 
	
	for j in 1:n_steps
		# println(j)
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

			if length(adjacent_triangles) > 0
				grad = point_gradient(i, points, triangles, adjacent_triangles, im_rgb)
				reg = regularization(i, points, triangles, adjacent_triangles)
	
				points[:, i] = update_point(point, step_size, λ, grad, reg, width, height)
			end
		end
		
		# Add triangles if error larger than threshold
		if (j % split_every == 0) & (n_triangles < max_n_triangles)
			to_split = []
			for i in 1:n_triangles
				triangle = points[:, triangles[:, i]]
				rmtae = sqrt(triangle_error(triangle, im_rgb; mean = true))
				size = triangle_size(triangle)
				stretchedness = triangle_stretchedness(triangle)

				if (rmtae > τ) & (size > min_triangle_size) & (stretchedness < max_triangle_stretchedness)
					n_neigh = [n_neighbors[p] for p in triangles[:, i]]
					if maximum(n_neigh) < 100
						# println("Split triangle")
						push!(to_split, i)
					end
				end
			end

			for i in to_split
				points, triangles = split_triangle(triangles, points, i)
				neighbors, neighbor_start_index, n_neighbors = generate_neighbors_lists(points, triangles)
			end
		end

		# Edge collapse of small triangles
		# Prevent vertices collapsing onto each other
		remove_edges = []
		if j % split_every == 0
			for i in 1:n_triangles
				triangle = points[:, triangles[:, i]]
				area = triangle_area(triangle)
				if area < min_area
					# println("Edge collapse")
					p1, p2, p3 = triangles[:, i]
					x1, x2, x3 = triangle[1, :]
					y1, y2, y3 = triangle[2, :]
					
					s1 = sqrt((x1-x2)^2 + (y1-y2)^2)
					s2 = sqrt((x2-x3)^2 + (y2-y3)^2)
					s3 = sqrt((x3-x1)^2 + (y3-y1)^2)
					smallest_edge = findfirst(min(s1, s2, s3) .== [s1, s2, s3])
					if smallest_edge == 1
						push!(remove_edges, p1, p2)
					elseif smallest_edge == 2
						push!(remove_edges, p2, p3)
					else
						push!(remove_edges, p3, p1)
					end
				end
			end
			# Remove points that belong to 2 triangles to be collapsed
			remove_points = [pt for (pt, cnt) in countmap(remove_edges) if cnt > 1]
			for i in remove_points
				# println("Remove point")
				points, triangles = remove_point(triangles, points, i, neighbors, neighbor_start_index, n_neighbors)
				neighbors, neighbor_start_index, n_neighbors = generate_neighbors_lists(points, triangles)
			end
		end
		
		# Edge flips to keep triangles as equi-angular as possible
		# Whenever the sum of the two angles that are opposite to an edge exceeds 180◦, an edge flip is performed
		if j % split_every == 0
			# Two triangles are candidates if they share two points
			while true
				common_points, opposite_points, (t1, t2) = find_flippable_edges(points, triangles)
				if common_points[1] == -1
					# No common points
					break
				elseif common_points[1] == -3
					# Remove duplicate
					# println("Removed triangle ", t2, " ", size(triangles)[2])
					n_triangles = size(triangles)[2]
					keep_triangles = [k for k in 1:n_triangles if k != t2]
					triangles = triangles[:, keep_triangles]
				else
					triangles[:, t1] = [common_points[1], opposite_points[1], 
								    opposite_points[2]]
					triangles[:, t2] = [common_points[2], opposite_points[1], 
									opposite_points[2]]
				end
			end
		end
		neighbors, neighbor_start_index, n_neighbors = generate_neighbors_lists(points, triangles)

		# save image right before splitting
		if (j+1) % split_every == 0
			draw_image(triangles, points, im_rgb, width, height, 2, j)
		end
	end
	
	points, triangles
end

# ╔═╡ 21c14a46-13c8-11eb-2780-2f4d79801128
im = load("vogels/gaai.jpeg")

# ╔═╡ c1549940-2765-11eb-32a2-3ff9fbf5e4e4
begin	
	height, width = size(im)
	im_rgb = real.(channelview(im))
	im_gray = gray.(float(Gray.(im)))
	
	step_size = 5.
	n_steps = 249
	λ = 0.002 # regularization size
	τ = 0.1 # error threshold to split triangle
	split_every = 20 # split triangles every ... steps
	min_triangle_size = 1/30 * min(height, width) # only split triangle if not too small
	max_triangle_stretchedness = 50. # don't want too narrow triangles
	max_n_triangles = 5000 # stop splitting if reached
	min_area = 1/10000 * width * height # collapse edge if area is less than
	
	# Generate initial grid
	# points, triangles = generate_regular_grid(width, height, 5, 5)
	points, triangles = generate_importance_grid(im_gray, width, height, 40)

 	neighbors, neighbor_start_index, n_neighbors = generate_neighbors_lists(points, triangles)
	
	# Perform gradient descent
	points, triangles = gradient_descent(points, triangles, neighbors,
										 neighbor_start_index, n_neighbors,
										 im_rgb, width, height,
										 n_steps, step_size, λ, τ, split_every,
										 max_n_triangles, min_triangle_size,
										 max_triangle_stretchedness, min_area)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Images = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
Luxor = "ae8d54c2-7ccd-5906-9d76-62fc9837b5bc"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
VoronoiDelaunay = "72f80fcb-8c52-57d9-aff0-40c1a3526986"

[compat]
Images = "~0.25.2"
Luxor = "~3.2.0"
StatsBase = "~0.33.16"
VoronoiDelaunay = "~0.4.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.1"
manifest_format = "2.0"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "6f1d9bc1c08f9f4a8fa92e3ea3cb50153a1b40d4"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.1.0"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "cf6875678085aed97f52bfc493baaebeb6d40bcb"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.5"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "d0b3f8b4ad16cb0a2988c6788646a5e6a17b6b1b"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.0.5"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9950387274246d08af38f6eef8cb5480862a435f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.14.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "75479b7df4167267d75294d14b58244695beb2ac"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.2"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "3f1f500312161f1ae067abe07d13b40f78f32e07"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.8"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "b153278a25dd42c65abbf4e62344f9d22e59191b"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.43.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[deps.CoordinateTransformations]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "681ea870b918e7cff7111da58791d7f718067a19"
uuid = "150eb455-5306-5404-9cee-2592286d6298"
version = "0.6.2"

[[deps.CustomUnitRanges]]
git-tree-sha1 = "1a3f97f907e6dd8983b744d2642651bb162a3f7a"
uuid = "dc8bdbbb-1ca9-579f-8c36-e416f6a65cce"
version = "1.0.2"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "cc1a8e22627f33c789ab60b36a9132ac050bbf75"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.12"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[deps.FFTViews]]
deps = ["CustomUnitRanges", "FFTW"]
git-tree-sha1 = "cbdf14d1e8c7c8aacbe8b19862e0179fd08321c2"
uuid = "4f61f5a4-77b1-5117-aa51-3ab5ef4ef0cd"
version = "0.3.2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "505876577b5481e50d089c1c68899dfb6faebc62"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.6"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "9267e5f50b0e12fdfd5a2455534345c4cf2c7f7a"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.14.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.GeometricalPredicates]]
git-tree-sha1 = "527d55e28ff359029d8f72d77c0bdcaf28793079"
uuid = "fd0ad045-b25c-564e-8f9c-8ef5c5f21267"
version = "0.4.1"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "78e2c69783c9753a91cdae88a8d432be85a2ab5e"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "1c5a84319923bea76fa145d49e93aa4394c73fc2"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.1"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "57c021de207e234108a6f1454003120a1bf350c4"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.6.0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "c54b581a83008dc7f292e205f4c409ab5caa0f04"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.10"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "b51bb8cae22c66d0f6357e3bcb6363145ef20835"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.5"

[[deps.ImageContrastAdjustment]]
deps = ["ImageCore", "ImageTransformations", "Parameters"]
git-tree-sha1 = "0d75cafa80cf22026cea21a8e6cf965295003edc"
uuid = "f332f351-ec65-5f6a-b3d1-319c6670881a"
version = "0.3.10"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "9a5c62f231e5bba35695a20988fc7cd6de7eeb5a"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.3"

[[deps.ImageDistances]]
deps = ["Distances", "ImageCore", "ImageMorphology", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "7a20463713d239a19cbad3f6991e404aca876bda"
uuid = "51556ac3-7006-55f5-8cb3-34580c88182d"
version = "0.2.15"

[[deps.ImageFiltering]]
deps = ["CatIndices", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageBase", "ImageCore", "LinearAlgebra", "OffsetArrays", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "15bd05c1c0d5dbb32a9a3d7e0ad2d50dd6167189"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.7.1"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "539682309e12265fbe75de8d83560c307af975bd"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.2"

[[deps.ImageMagick]]
deps = ["FileIO", "ImageCore", "ImageMagick_jll", "InteractiveUtils", "Libdl", "Pkg", "Random"]
git-tree-sha1 = "5bc1cb62e0c5f1005868358db0692c994c3a13c6"
uuid = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
version = "1.2.1"

[[deps.ImageMagick_jll]]
deps = ["Artifacts", "Ghostscript_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f025b79883f361fa1bd80ad132773161d231fd9f"
uuid = "c73af94c-d91f-53ed-93a7-00f77d67a9d7"
version = "6.9.12+2"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "36cbaebed194b292590cba2593da27b34763804a"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.8"

[[deps.ImageMorphology]]
deps = ["ImageCore", "LinearAlgebra", "Requires", "TiledIteration"]
git-tree-sha1 = "7668b123ecfd39a6ae3fc31c532b588999bdc166"
uuid = "787d08f9-d448-5407-9aad-5290dd7ab264"
version = "0.3.1"

[[deps.ImageQualityIndexes]]
deps = ["ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "OffsetArrays", "Statistics"]
git-tree-sha1 = "1d2d73b14198d10f7f12bf7f8481fd4b3ff5cd61"
uuid = "2996bd0c-7a13-11e9-2da2-2f5ce47296a9"
version = "0.3.0"

[[deps.ImageSegmentation]]
deps = ["Clustering", "DataStructures", "Distances", "Graphs", "ImageCore", "ImageFiltering", "ImageMorphology", "LinearAlgebra", "MetaGraphs", "RegionTrees", "SimpleWeightedGraphs", "StaticArrays", "Statistics"]
git-tree-sha1 = "36832067ea220818d105d718527d6ed02385bf22"
uuid = "80713f31-8817-5129-9cf8-209ff8fb23e1"
version = "1.7.0"

[[deps.ImageShow]]
deps = ["Base64", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "25f7784b067f699ae4e4cb820465c174f7022972"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.4"

[[deps.ImageTransformations]]
deps = ["AxisAlgorithms", "ColorVectorSpace", "CoordinateTransformations", "ImageBase", "ImageCore", "Interpolations", "OffsetArrays", "Rotations", "StaticArrays"]
git-tree-sha1 = "42fe8de1fe1f80dab37a39d391b6301f7aeaa7b8"
uuid = "02fcd773-0e25-5acc-982a-7f6622650795"
version = "0.9.4"

[[deps.Images]]
deps = ["Base64", "FileIO", "Graphics", "ImageAxes", "ImageBase", "ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "ImageIO", "ImageMagick", "ImageMetadata", "ImageMorphology", "ImageQualityIndexes", "ImageSegmentation", "ImageShow", "ImageTransformations", "IndirectArrays", "IntegralArrays", "Random", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "TiledIteration"]
git-tree-sha1 = "03d1301b7ec885b266c0f816f338368c6c0b81bd"
uuid = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
version = "0.25.2"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "87f7662e03a649cffa2e05bf19c303e168732d3e"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.2+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[deps.IntegralArrays]]
deps = ["ColorTypes", "FixedPointNumbers", "IntervalSets"]
git-tree-sha1 = "509075560b9fce23fdb3ccb4cc97935f11a43aa0"
uuid = "1d092043-8f09-5a30-832f-7509e371ab51"
version = "0.1.4"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "b7bc05649af456efc75d178846f47006c2c4c3c7"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.6"

[[deps.IntervalSets]]
deps = ["Dates", "Statistics"]
git-tree-sha1 = "eb381d885e30ef859068fce929371a8a5d06a914"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.6.1"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "91b5dcf362c5add98049e6c29ee756910b03051d"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.3"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "Pkg", "Printf", "Reexport", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "81b9477b49402b47fbe7f7ae0b252077f53e4a08"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.22"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "a77b273f1ddec645d1b7c4fd5fb98c8f90ad10a5"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.1"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.Juno]]
deps = ["Base64", "Logging", "Media", "Profile"]
git-tree-sha1 = "07cb43290a840908a771552911a6274bc6c072c7"
uuid = "e5e0dc1b-0480-54bc-9374-aad01c23163d"
version = "0.8.4"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Librsvg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pango_jll", "Pkg", "gdk_pixbuf_jll"]
git-tree-sha1 = "25d5e6b4eb3558613ace1c67d6a871420bfca527"
uuid = "925c91fb-5dd6-59dd-8e8c-345e74382d89"
version = "2.52.4+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "c9551dd26e31ab17b86cbd00c2ede019c08758eb"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "76c987446e8d555677f064aaac1145c4c17662f8"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.14"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.Luxor]]
deps = ["Base64", "Cairo", "Colors", "Dates", "FFMPEG", "FileIO", "Juno", "LaTeXStrings", "Random", "Requires", "Rsvg"]
git-tree-sha1 = "156958d51d9f758dc5a00dcc6da4f61cacf579ed"
uuid = "ae8d54c2-7ccd-5906-9d76-62fc9837b5bc"
version = "3.2.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "e595b205efd49508358f7dc670a940c790204629"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.0.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Media]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "75a54abd10709c01f1b86b84ec225d26e840ed58"
uuid = "e89f7d12-3494-54d1-8411-f7d8b9ae1f27"
version = "0.5.0"

[[deps.MetaGraphs]]
deps = ["Graphs", "JLD2", "Random"]
git-tree-sha1 = "2af69ff3c024d13bde52b34a2a7d6887d4e7b438"
uuid = "626554b9-1ddb-594c-aa3c-2596fe9399a5"
version = "0.7.1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "b34e3bc3ca7c94914418637cb10cc4d1d80d877d"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.3"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NaNMath]]
git-tree-sha1 = "b086b7ea07f8e38cf122f5016af580881ac914fe"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.7"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "ded92de95031d4a8c61dfb6ba9adb6f1d8016ddd"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.10"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore"]
git-tree-sha1 = "18efc06f6ec36a8b801b23f076e3c6ac7c3bf153"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "043017e0bdeff61cfbb7afeb558ab29536bbb5ed"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.8"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "923319661e9a22712f24596ce81c54fc0366f304"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.1+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ab05aa4cc89736e95915b01e7279e61b1bfe33b8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.14+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "e925a64b8585aa9f4e3047b8d2cdc3f0e79fd4e4"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.3.16"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "03a7a85b76381a3d04c7a1656039197e70eda03d"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.11"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a121dfbba67c94a5bec9dde613c3d0cbcf3a12b"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.50.3+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "a7a7e1a88853564e551e4eba8650f8c38df79b37"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.1.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[deps.Quaternions]]
deps = ["DualNumbers", "LinearAlgebra", "Random"]
git-tree-sha1 = "b327e4db3f2202a4efafe7569fcbe409106a1f75"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.5.6"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "dc84268fe0e3335a62e315a3a7cf2afa7178a734"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.3"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RegionTrees]]
deps = ["IterTools", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "4618ed0da7a251c7f92e869ae1a19c74a7d2a7f9"
uuid = "dee08c22-ab7f-5625-9660-a9af2021b33f"
version = "0.3.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rotations]]
deps = ["LinearAlgebra", "Quaternions", "Random", "StaticArrays", "Statistics"]
git-tree-sha1 = "3177100077c68060d63dd71aec209373c3ec339b"
uuid = "6038ab10-8711-5258-84ad-4b1120ba62dc"
version = "1.3.1"

[[deps.Rsvg]]
deps = ["Cairo", "Glib_jll", "Librsvg_jll"]
git-tree-sha1 = "3d3dc66eb46568fb3a5259034bfc752a0eb0c686"
uuid = "c4c386cf-5103-5370-be45-f3a111cca3b8"
version = "1.0.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SimpleWeightedGraphs]]
deps = ["Graphs", "LinearAlgebra", "Markdown", "SparseArrays", "Test"]
git-tree-sha1 = "a6f404cc44d3d3b28c793ec0eb59af709d827e4e"
uuid = "47aef6b3-ad0c-573a-a1e2-d07658019622"
version = "1.2.1"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "8fb59825be681d451c246a795117f317ecbcaa28"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.2"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "5ba658aeecaaf96923dce0da9e703bd1fe7666f9"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.4"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "cd56bf18ed715e8b09f06ef8c6b781e6cdc49911"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c82aaa13b44ea00134f8c9c89819477bd3986ecd"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.3.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "f90022b44b7bf97952756a6b6737d1a0024a3233"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.5.5"

[[deps.TiledIteration]]
deps = ["OffsetArrays"]
git-tree-sha1 = "5683455224ba92ef59db72d10690690f4a8dc297"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.3.1"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.VoronoiDelaunay]]
deps = ["Colors", "GeometricalPredicates", "Random"]
git-tree-sha1 = "ed19f55808fb99951d36e8616a95fc9d94045466"
uuid = "72f80fcb-8c52-57d9-aff0-40c1a3526986"
version = "0.4.1"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.gdk_pixbuf_jll]]
deps = ["Artifacts", "Glib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg", "Xorg_libX11_jll", "libpng_jll"]
git-tree-sha1 = "c23323cd30d60941f8c68419a70905d9bdd92808"
uuid = "da03df04-f53b-5353-a52f-6a8b0620ced0"
version = "2.42.6+1"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "78736dab31ae7a53540a6b752efc61f77b304c5b"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.8.6+1"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"
"""

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
# ╠═ea76e3f6-6ef6-4375-a635-b8720fab5602
# ╠═66877a86-2985-11eb-3e14-439d0b057cbb
# ╠═3381b436-485d-4ec6-b80c-aa9ddad6624d
# ╠═3480aa2c-2e71-4d62-ad15-fc38b17184ba
# ╠═562a1e58-118b-11eb-28a1-c576d975bb6d
# ╠═c86c8d8c-2433-11eb-3c26-e51f78d97a9c
# ╠═fa734135-983f-4ea2-b3f3-fee90a97dc26
# ╠═21c14a46-13c8-11eb-2780-2f4d79801128
# ╠═c1549940-2765-11eb-32a2-3ff9fbf5e4e4
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
