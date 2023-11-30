import requests
import time_series_object
import pickle
import numpy as np

url = 'https://pp2.xdx.gg/chart/1/kr/'

classifications = ["Top", "Jungle", "Mid", "Bot", "Support"]

matchIds = [6813858262, 6814040342, 6814143598, 6814217313, 6814323205,
            6813221750, 6813482960, 6813980049, 6814137112, 6814323205,
            6814143598, 6814217313, 6814343606, 6814374790, 6814408275,
            6812076485, 6812290137, 6812333300, 6812425399, 6812474348,
            6801346096, 6801671645, 6801754920, 6801852771, 6801936316,
            6807254523, 6807331284, 6807506981, 6807538968, 6807565769,
            6810562774, 6810629002, 6810661341, 6810707508, 6810752814,
            6808010035, 6808943107, 6808990477, 6809040222, 6809069506,
            6791083406, 6795865188, 6796230714, 6798047794, 6798136097,
            6800545492, 6800616113, 6800632265, 6802074557, 6802161135,
            6810782706, 6810813903, 6813063887, 6813363970, 6813550873,
            6807396168, 6807814420, 6809032127, 6814244647, 6814285253,
            6804149595, 6804221282, 6804304808, 6804330816, 6804368328,
            6814456875, 6814481307, 6814498895, 6814528344, 6814591191,
            6812937886, 6812981946, 6813023729, 6813144292, 6813180693,
            6798357248, 6803394239, 6804023578, 6804295073, 6805559175,
            6809441262, 6809522293, 6814217313, 6814276588, 6814352940,
            6815223311, 6815277673, 6815331335, 6815396014, 6815466975,
            6814855118, 6814907472, 6816153758, 6816291371, 6816355779,
            6810441227, 6810534085, 6810573686, 6813591892, 6813674496,
            6810079063, 6810567904, 6810697452, 6810735278, 6812422734,
            6814323205, 6815088425, 6816155834, 6816243311, 6816448694,
            6814939372, 6815405551, 6815512740, 6815575423, 6815689976,
            6809501654, 6810276219, 6810744058, 6810755944, 6810778567,
            6811600474, 6812292954, 6812497895, 6814276053, 6814343606,
            6808796653, 6811791421, 6812108406, 6812368316, 6812827493,
            6799645597, 6800073444, 6800361766, 6800452263, 6800534856,
            6812292954, 6812333300, 6812393139, 6813369653, 6813479666,
            6810782706, 6810813903, 6813063887, 6813363970, 6813550873,
            6811600474, 6811647827, 6812877200, 6813363970, 6813539693]

time_series_objects = []

for matchId in matchIds:
    response = requests.get(url + str(matchId))

    if response.status_code == 200:
        data = response.json()

        for idx, (gold, xp, cs) in enumerate(zip(data["gold"], data["xp"], data["cs"])):
            current_time_series = time_series_object.TimeSeriesObject(
                classification=classifications[idx % len(classifications)],
                gold=np.array(gold),
                xp=np.array(xp),
                cs=np.array(cs))
            time_series_objects.append(current_time_series)
    else:
        print('Failed to fetch data:', response.status_code)

with open('1500_data_objects.pkl', 'wb') as file:
    pickle.dump(time_series_objects, file)