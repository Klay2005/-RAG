import os
from nba_api.stats.static import players
from nba_api.stats.endpoints import playercareerstats

def get_nba_hero_data(player_name, folder="nba_docs"):
    """抓取球星生涯数据并转成 RAG 友好的文本"""
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 1. 查找球员 ID
    nba_players = players.find_players_by_full_name(player_name)
    if not nba_players:
        print(f"找不到球员: {player_name}")
        return
    
    player_id = nba_players[0]['id']
    full_name = nba_players[0]['full_name']
    
    # 2. 获取生涯常规赛统计数据
    career = playercareerstats.PlayerCareerStats(player_id=player_id)
    df = career.get_data_frames()[0] # 获取数据表
    
    # 3. 将数据转化为自然语言描述
    text_content = f"### {full_name} 职业生涯详细数据报告\n\n"
    
    for _, row in df.iterrows():
        season = row['SEASON_ID']
        team = row['TEAM_ABBREVIATION']
        pts = row['PTS']
        reb = row['REB']
        ast = row['AST']
        gp = row['GP'] # 出场次数
        
        # 关键逻辑：将结构化数据转为描述性文字
        line = f"在 {season} 赛季，{full_name} 效力于 {team} 队。该赛季他共出战 {gp} 场比赛，" \
               f"场均（总计）贡献 {pts} 分、{reb} 个篮板和 {ast} 次助攻。\n"
        text_content += line

    # 4. 保存为 TXT 文件
    file_path = os.path.join(folder, f"{full_name.replace(' ', '_')}_stats.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text_content)
    
    print(f"✅ 已生成 {full_name} 的 RAG 知识库文件：{file_path}")

# --- 测试运行 ---
if __name__ == "__main__":
    stars = ["LeBron James", "Stephen Curry", "Kevin Durant","Kyrie Irving","Kobe Bryant",'Luka Doncic','Devin Booker','Yao Ming','Joel Embiid','Anthony Davis','Jimmy Butler','James Harden','Kawhi Leonard']
    for star in stars:
        get_nba_hero_data(star)