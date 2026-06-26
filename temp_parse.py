import json
with open("results/clap_kruskal_dunn.json", "r") as f:
    data = json.load(f)
print("H global:", data["overall"]["h_statistic"])
print("p global:", data["overall"]["p_value"])
print("eta squared:", data["overall"]["eta_squared"])
for d in data["overall"]["dunn"]:
    print(f'{d["model_a"]} & {d["model_b"]} & {d["z"]:.2f} & {d["p_value"]:.1e} & {d["p_bonferroni"]:.1e} \\\\')
for genre, gdata in data["by_genre"].items():
    print(f'{genre.capitalize()} & {gdata["h_statistic"]:.2f} & {gdata["p_value"]:.1e} \\\\')
