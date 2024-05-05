using CSV, DataFrames, Plots, StatsBase

Data = CSV.File("C:/Users/adria/Desktop/Programowanie/python/6_semester/mathematics_for_industry/old/stops.txt") |> DataFrame

stacje = Data
xs = stacje.stop_lat
ys = stacje.stop_lon
##
scatter(ys, xs, astpect_ratio=1)
##
#tr = filter(row->row.route_id=="ZKA TLK", trips)
##
trips = CSV.File("C:/Users/adria/Desktop/Programowanie/python/6_semester/mathematics_for_industry/old/trips.txt") |> DataFrame
utrips = [["IC", "ZKA IC"], ["TLK", "ZKA TLK"], ["EIC"], ["EIP"]]
st = CSV.File("C:/Users/adria/Desktop/Programowanie/python/6_semester/mathematics_for_industry/old/stop_times.txt") |> DataFrame

RX = Vector{Vector{Vector{Float64}}}(undef, length(utrips))
RY = Vector{Vector{Vector{Float64}}}(undef, length(utrips))
for (k, rid) in enumerate(utrips)
    tr = filter(row->row.route_id ∈ rid, trips)

    ICtrips = tr.trip_id

    st = CSV.File("C:/Users/adria/Desktop/Programowanie/python/6_semester/mathematics_for_industry/old/stop_times.txt") |> DataFrame

    filter!(row->row.trip_id ∈ ICtrips, st)
    rt = unique(st, :trip_id)
    s = groupby(st, :trip_id)

    D = rt.trip_id
    d1 = findfirst((x)->parse(Int64, x[9:10])>=11, D)
    d2 = findfirst((x)->parse(Int64, x[9:10])>=12, D)-1


    X = Vector{Vector{Float64}}(undef, d2-d1+1)
    Y = Vector{Vector{Float64}}(undef, d2-d1+1)
    for (j, trid) in enumerate(rt.trip_id[d1:d2])
        g = s[j].stop_id

        x = Vector{Float64}(undef, length(g))
        y = Vector{Float64}(undef, length(g))
        for (i, stop) in enumerate(g)
            a = filter(row->row.stop_id==stop, stacje)
            x[i] = a.stop_lat[1]
            y[i] = a.stop_lon[1]
        end
        X[j] = x
        Y[j] = y
    end
    RX[k] = X
    RY[k] = Y
end
##
#img = load("Poland_map_flag.png")
##
scatter(ys, xs, legend=false, aspect_ratio=1)
plot(aspect_ratio=1, legend=false)
# plot!(bord[:,1], bord[:,2], c=:black)
scatter!(ys, xs, ms=2.3, markerstrokewidth=0, c=:gray)
for j in eachindex(RX)
    X = RX[j]
    Y = RY[j]
    for i in eachindex(X)
        plot!(Y[i], X[i], c=j, lw=2, alpha=0.05)
    end
end
plot!()
##
RXA = Vector{Vector{Vector{Float64}}}(undef, length(RX))
RYA = Vector{Vector{Vector{Float64}}}(undef, length(RX))
for j in eachindex(RX)
    XA = Vector{Float64}[]
    YA = Vector{Float64}[]
    X = RX[j]
    Y = RY[j]
    for i in eachindex(X)
        for k in eachindex(X[i])[1:end-1]
            push!(XA, [X[i][k], X[i][k+1]])
            push!(YA, [Y[i][k], Y[i][k+1]])
        end
    end
    RXA[j] = XA
    RYA[j] = YA
end
##
c = countmap.(RXA)
uRXA = unique.(RXA)
uRYA = unique.(RYA)
intIC = [c[1][x] for x in uRXA[1]]
intTLK = [c[2][x] for x in uRXA[2]]
intEIC = [c[3][x] for x in uRXA[3]]
intEIP = [c[4][x] for x in uRXA[4]]
intR = [intIC, intTLK, intEIC, intEIP]
##
findall(x->x==uRXA[1][26], RXA[2])
##
colors = [:lightskyblue,:seagreen,:tan1,:indianred]
labels=["IC","TLK","EIC","EIP"]

p = plot( ylim=(48.8,55.2), xlim=(13.7,24.3),dpi=1000,
legendfontsize=15,legend=:bottomleft,size=(900,800),ascpect_ratio=1.5,
tickfontsize=15,labelfontsize=15,xlabel="długość geograficzna[∘]",ylabel="szerekość geograficzna[∘]",legend_font_pointsize=4)

# p = plot!(bord[:,1], bord[:,2], c=:black,label=false)
for i in 1:4
    intense = intR[i]
    m = maximum(intense)
    p = plot!([0,1],[0,1],c=colors[i],label=labels[i],lw=3)
    for j in eachindex(intense)
        p = plot!(uRYA[i][j], uRXA[i][j], label=false, c=colors[i],
        lw=0.7*intense[j]+1, alpha=0.3+intense[j]/m*0.7)
    end

end
plot!(title="Sieć połączeń PKP InterCity",titlefontsize=20)
p = scatter!(ys, xs, label="stacje",markerstrokewidth=0,c=:gray60,ms=2.5)


savefig(p, "graf1")







##
sum([length(unique(sort.(uRXA[i]))) for i in 4])
##
unique(filter(row->row.route_id ∈ ["IC", "ZKA  IC"], trips), :trip_id)
##
groupby(filter(row->row.trip_id[1:10]=="2024-03-11", trips), :route_id)
##
A = Vector{Vector}(undef, 6)
for (k, i) in enumerate(["IC", "TLK", "EIC", "EIP", "ZKA IC", "ZKA TLK"])
    A[k] = unique(filter(row->(row.route_id==i && row.trip_id[1:10]=="2024-03-11"), trips), :trip_short_name).trip_short_name
end
length.(A)
##
findmax.(intR)
##
filter(row->row.stop_lat==uRXA[1][26][1], stacje)
filter(row->row.stop_lat==uRXA[1][26][2], stacje)
##
filter(row->row.stop_lat==uRXA[2][62][1], stacje)
filter(row->row.stop_lat==uRXA[2][62][2], stacje)
##
filter(row->row.stop_lat==uRXA[3][1][1], stacje)
filter(row->row.stop_lat==uRXA[3][1][2], stacje)
##
filter(row->row.stop_lat==uRXA[4][1][1], stacje)
filter(row->row.stop_lat==uRXA[4][1][2], stacje)
##
RA = Vector{Vector{Vector{Float64}}}(undef, length(RX))
for j in eachindex(RX)
    XA = Vector{Float64}[]
    X = RX[j]
    Y = RY[j]
    for i in eachindex(X)
        for k in eachindex(X[i])
            push!(XA, [X[i][k], Y[i][k]])
        end
    end
    RA[j] = XA
end
##
RXB = Vector{Vector{Float64}}(undef, 4)
for i in 1:4
    E = Vector{Float64}(undef, length(RXA[i])*2)
    for j in eachindex(RXA[i])
        E[2j-1] = RXA[i][j][1]
        E[2j] = RXA[i][j][2]
    end
    RXB[i] = E
end
##
v = countmap.(RXB)
n = findmax.(v)
##
filter(row->row.stop_lat==n[1][2], stacje)
filter(row->row.stop_lat==n[2][2], stacje)
filter(row->row.stop_lat==n[3][2], stacje)
filter(row->row.stop_lat==n[4][2], stacje)
##
w = countmap(filter(row->row.trip_id[1:10]=="2024-03-11", st).stop_id)
findmax(w)

filter(row->row.stop_id==38653, stacje)
