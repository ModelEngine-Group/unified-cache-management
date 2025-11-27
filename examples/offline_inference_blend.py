import contextlib
import json
import os
import re
import time
import csv
from dataclasses import asdict

from vllm.v1.metrics.reader import Counter, Gauge, Histogram, Vector
from tqdm import tqdm
import random
random.seed(0)

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

from ucm.logger import init_logger
from vllm.inputs import TokensPrompt

import sys
logger = init_logger(__name__)

model = ""
data_dir = ""
tokenizer = None
# 28705 is the token id for <space> char in llama model
# 151643 is the pad token id in qwen model
chunk_end_token_id = -1
chunk_pad_token_id = -1
block_size = 64

def setup_environment_variables():
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["PYTHONHASHSEED"] = "123456"

    global model, data_dir, tokenizer, chunk_end_token_id, chunk_pad_token_id
    model = os.getenv("MODEL_PATH", "/home/models/mistralai/Mistral-7B-Instruct-v0.2")
    if not os.path.isdir(model):
        model = input("Enter path to model, e.g./home/models/mistralai/Mistral-7B-Instruct-v0.2: ")
        if not os.path.isdir(model):
            print("Exiting. Incorrect model_path")
            sys.exit(1)


    data_dir = os.getenv("DATA_DIR", "/home/data/kv_cache")
    if not os.path.isdir(data_dir):
        data_dir = input(
            "Enter the directory for UCMStore to save kv cache, e.g. /home/data/kv_cache: "
        )
        create = input(f"Directory {data_dir} dose not exist. Create it? (Y/n): ")
        if create.lower() == "y":
            os.makedirs(data_dir, exist_ok=True)
        else:
            print("Exiting. Directory not created.")
            sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(model, use_chat_template=True)
    # as for Qwen model, use pad_token_id for padding block
    # as for Llama model, current use unk_token for  padding block
    chunk_pad_token_id = tokenizer.encode("▁", add_special_tokens=False)[0]
    chunk_end_token_id = chunk_pad_token_id

    if tokenizer.pad_token_id is not None:
        chunk_pad_token_id = tokenizer.pad_token_id
        chunk_end_token_id = tokenizer.pad_token_id


@contextlib.contextmanager
def build_llm_with_uc(module_path: str, name: str, model: str):
    ktc = KVTransferConfig(
        kv_connector=name,
        kv_connector_module_path=module_path,
        kv_role="kv_both",
        kv_connector_extra_config={
            "ucm_connector_name": "UcmNfsStore",
            "ucm_connector_config": {
                "storage_backends": data_dir,
                "kv_block_size": 33554432,
            },
            "use_layerwise": True,
            "ucm_sparse_config": {
                "Blend": {
                    "chunk_end_token_id": chunk_end_token_id,
                    "compute_meta": {
                        "model.layers.1.self_attn.attn": {
                            "ratio": 0.2,
                        },
                    }
                }
            },
        },
    )

    llm_args = EngineArgs(
        model=model,
        enforce_eager=True,
        kv_transfer_config=ktc,
        max_model_len=16384 * 2,
        max_num_batched_tokens=16384 * 2,
        gpu_memory_utilization=0.8,
        block_size=block_size,
        enable_prefix_caching=False,
        distributed_executor_backend="mp",
        tensor_parallel_size=1,
        trust_remote_code=True,
    )

    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        logger.info("LLM engine is exiting.")


def get_output(
    llm: LLM,
    prompt,
    sampling_params: SamplingParams,
):
    start = time.time()
    outputs = llm.generate(prompt, sampling_params)
    print("-" * 50)
    generated_text = None
    for output in outputs:
        generated_text = output.outputs[0].text
    e2e_time = time.time() - start
    print("-" * 50)
    return e2e_time, generated_text


def pad_rag_chunks(token_ids, block_size, pad_id, end_id):
    """
    pad token_ids with pad_id and end up with end_id
    """
    # assert pad_id != end_id
    remainder = len(token_ids) % block_size

    if remainder == 0 and token_ids[-1] in [pad_id, end_id]:
        # no need to pad
        token_ids[-1] = end_id
        return token_ids

    pad_len = block_size - remainder - 1
    padded = token_ids + [pad_id] * pad_len + [end_id]
    return padded


def main():
    module_path = "ucm.integration.vllm.uc_connector"
    name = "UnifiedCacheConnectorV1"

    setup_environment_variables()

    with build_llm_with_uc(module_path, name, model) as llm:
        prefill_sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=1)
        sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=128)
        # choose one data row in LongBenchV1 (wikimqa)
        dataset_row = {"input": "Where was the wife of Francis I Rákóczi born?",
                       "context": "Passage 1:\nWaldrada of Lotharingia\nWaldrada was the mistress, and later the wife, of Lothair II of Lotharingia.\n\nBiography\nWaldrada's family origin is uncertain. The prolific 19th-century French writer Baron Ernouf suggested that Waldrada was of noble Gallo-Roman descent, sister of Thietgaud, the bishop of Trier, and niece of Gunther, archbishop of Cologne. However, these suggestions are not supported by any evidence, and more recent studies have instead suggested she was of relatively undistinguished social origins, though still from an aristocratic milieu.\nThe Vita Sancti Deicoli states that Waldrada was related to Eberhard II, Count of Nordgau (included Strasbourg) and the family of Etichonids, though this is a late 10th-century source and so may not be entirely reliable on this question.In 855 the Carolingian king Lothar II married Teutberga, a Carolingian aristocrat and the daughter of Bosonid Boso the Elder. The marriage was arranged by Lothar's father Lothar I for political reasons. It is very probable that Waldrada was already Lothar II's mistress at this time.Teutberga was allegedly not capable of bearing children and Lothar's reign was chiefly occupied by his efforts to obtain an annulment of their marriage, and his relations with his uncles Charles the Bald and Louis the German were influenced by his desire to obtain their support for this endeavour. Lothair, whose desire for annulment was arguably prompted by his affection for Waldrada, put away Teutberga. However, Hucbert took up arms on his sister's behalf, and after she had submitted successfully to the ordeal of water, Lothair was compelled to restore her in 858. Still pursuing his purpose, he won the support of his brother, Emperor Louis II, by a cession of lands and obtained the consent of the local clergy to the annulment and to his marriage with Waldrada, which took place in 862. However, Pope Nicholas I was suspicious of this and sent legates to investigate at the Council of Metz in 863. The Council found in favour of Lothair's divorce, which led to rumours that the papal legates may have bribed and thus meant that Nicholas order Lothair to take Teutberga back or face excommunication. \nWith the support of Charles the Bald and Louis the German, Teutberga appealed the annulment to Pope Nicholas. Nicholas refused to recognize the annulment and excommunicated Waldrada in 866, forcing Lothair to abandon Waldrada in favour of Teutberga. Lothair accepted this begrudgingly for a time, but shortly afterward at the end of 867 Pope Nicholas I died. Thus, Lothair began to seek the permission of the newly appointed Pope Adrian II to again put Teutberga aside and marry Waldrada, riding to Rome to speak with him on the matter in 869. However, on his way home, Lothair died.\n\nChildren\nWaldrada and Lothair II had some sons and probably three daughters, all of whom were declared illegitimate:\n\nHugh (c. 855–895), Duke of Alsace (867–885)\nGisela (c. 865–908), who in 883 married Godfrey, the Viking leader ruling in Frisia, who was murdered in 885\nBertha (c. 863–925), who married Theobald of Arles (c. 854–895), count of Arles, nephew of Teutberga. They had two sons, Hugh of Italy and Boso of Tuscany. After Theobald's death, between 895 and 898 she married Adalbert II of Tuscany (c. 875–915) They had at least three children: Guy, who succeeded his father as count and duke of Lucca and margrave of Tuscany, Lambert succeeded his brother in 929, but lost the titles in 931 to his half-brother Boso of Tuscany, and Ermengard.\nErmengarde (d. 90?)\nOdo (d. c.879)\nPassage 2:\nFrancis I Rákóczi\nFrancis I Rákóczi (February 24, 1645, Gyulafehérvár, Transylvania – July 8, 1676, Zboró, Royal Hungary) was a Hungarian aristocrat, elected prince of Transylvania and father of Hungarian national hero Francis Rákóczi II.Francis Rákóczi was the son of George Rákóczi II, prince of Transylvania, and Sophia Báthory. He was elected prince by the Transylvanian Diet in 1652, during his father's life. However, because of the disastrous Polish campaign of 1657 and its consequences, the Ottoman Empire removed his father from the throne in 1660, and prohibited any Rákóczi to ascend the Transylvanian throne. This left Francis unable to come by his father's legacy; he therefore withdrew to his estates in Royal Hungary.\nNotably, the Rákóczi family was Calvinist, and they were staunch supporters of the Reformed Church in Hungary. However, Francis' mother, Sophia Báthory, had converted to Calvinism merely for the sake of her marriage. After her husband's death, she returned to Catholicism and supported the Counter Reformation. Francis Rákóczi also became a Catholic, thus acquiring favour with the Catholic Habsburg Court. His mother converted him to Catholicism. He was made a count in 1664.\nIn 1666 Francis married Jelena Zrinska (Hungarian: Zrínyi Ilona), a Croatian countess, and joined the Wesselényi conspiracy (Zrinski-Frankopan conspiracy in Croatia), one leader of which was Jelena's father, Petar Zrinski (Hungarian: Zrínyi Péter). Francis soon became the leader of the conspiracy, and, as a culmination of their anti-Habsburg stratagems, started an armed uprising of nobles in Upper Hungary, while the other conspirators were supposed to start the fight in Croatia. Due to poor organization and discord between the conspirators, however, the Austrian authorities were well informed; they quickly suppressed the Croatian branch of the revolt.\nWhen Rákóczi learned that Petar Zrinski had been captured by the Austrians, he laid down his arms and applied for mercy. All other leaders of the conspiracy were executed for high treason; Rákóczi, due to his mother's intervention, and for a ransom of 300,000 forints and several castles, was pardoned.\n\nIssue\nFrancis I had three  children:\n\nGyörgy (1667)\nJulianna Borbála (1672–1717), married Count Ferdinand Gobert von Aspremont-Lynden (1643-1708)\nFrancis Rákóczi II (1676–1735)Francis II was born only three months before his father's death. He led a rebellion against Austrian rule (Rákóczi's War of Independence) and died in exile.\nPassage 3:\nMary Fiennes (lady-in-waiting)\nMary Fiennes (1495–1531) was an English courtier. She was the wife of Henry Norris. Norris was executed for treason as one of the alleged lovers of her cousin, Anne Boleyn, the second wife of King Henry VIII of England. Mary lived for six years at the French court as a Maid of Honour to queens consort Mary Tudor, wife of Louis XII; and Claude of France, wife of Francis I.\n\nFamily and early years\nMary was born at Herstmonceux Castle in Sussex in 1495, the only daughter of Thomas Fiennes, 8th Baron Dacre and Anne Bourchier. By both her father and mother she was descended from Edward III. She had two younger brothers, Sir Thomas and John. Her mother was an elder half-sister of Elizabeth Howard and Lord Edmund Howard, making queen consorts Anne Boleyn and Catherine Howard a cousin of Mary. Her paternal grandmother, Alice FitzHugh, was sister to Elizabeth FitzHugh, grandmother of Catherine Parr, making her cousin to yet another queen consort of Henry VIII.\nIn 1514, Mary was appointed a Maid of Honour to Princess Mary Tudor and accompanied her to France when the latter married King Louis XII of France; afterwards she served in the capacity to Queen Mary's successor, Queen Claude, consort of the new king Francis I of France. Among her fellow Maids of Honour were her cousins, Mary (a mistress of Henry VIII) and Anne Boleyn.\n\nMarriage and issue\nIn 1520 upon her return to England, she married the courtier, Henry Norreys (1491 – 17 May 1536) of Yattendon in Berkshire, whom she had met that same year at the Field of the Cloth of Gold in France.\nNorris served King Henry VIII of England as a Gentleman of the Bedchamber, and was held in high favour by the King. He was later appointed Groom of the Stool and continued to enjoy the King's favour. According to biographer Eric Ives, Norris was \"perhaps the nearest thing Henry had to a friend.\" Norris had control of King Henry's Privy chamber.\nHenry and Mary had three children:\nEdward Norris (died 1529)\nHenry Norris, 1st Baron Norreys (c. 1525 – 1601), married Margaret Williams of Rycote, by whom he had issue.\nMary Norris, married firstly Sir George Carew, and secondly Sir Arthur Champernowne, by whom she had issue.\n\nDeath\nMary died in 1531, a year after her mother. Five years later her husband was attainted and executed for treason as one of the five alleged lovers of her cousin Queen Anne Boleyn, who herself was beheaded at the Tower of London on 19 May 1536.\nTheir four orphaned children were raised by Norris's brother Sir John Norris.\n\nAncestry\nPassage 4:\nAgatha (wife of Samuel of Bulgaria)\nAgatha (Bulgarian: Агата, Greek: Άγάθη; fl. late 10th century) was the wife of Emperor Samuel of Bulgaria.\n\nBiography\nAccording to a later addition to the history of the late-11th-century Byzantine historian John Skylitzes, Agatha was a captive from Larissa, and the daughter of the magnate of Dyrrhachium, John Chryselios. Skylitzes explicitly refers to her as the mother of Samuel's heir Gavril Radomir, which means that she was probably Samuel's wife. On the other hand, Skylitzes later mentions that Gavril Radomir himself also took a beautiful captive, named Irene, from Larissa as his wife. According to the editors of the Prosopographie der mittelbyzantinischen Zeit, this may have been a source of confusion for a later copyist, and Agatha's real origin was not Larissa, but Dyrrhachium. According to the same work, it is likely that she had died by ca. 998, when her father surrendered Dyrrhachium to the Byzantine emperor Basil II.Only two of Samuel's and Agatha's children are definitely known by name: Gavril Radomir and Miroslava. Two further, unnamed, daughters are mentioned in 1018, while Samuel is also recorded as having had a bastard son.Agatha is one of the central characters in Dimitar Talev's novel Samuil.\nPassage 5:\nEmpress Shōken\nEmpress Dowager Shōken (昭憲皇太后, Shōken-kōtaigō, 9 May 1849 – 9 April 1914), born Masako Ichijō (一条勝子, Ichijō Masako), was the wife of Emperor Meiji of Japan. She is also known under the technically incorrect name Empress Shōken (昭憲皇后, Shōken-kōgō). She was one of the founders of the Japanese Red Cross Society, whose charity work was known throughout the First Sino-Japanese War.\n\nEarly life\nLady Masako Ichijō was born on 9 May 1849, in Heian-kyō, Japan. She was the third daughter of Tadayoshi Ichijō, former Minister of the Left and head of the Fujiwara clan's Ichijō branch. Her adoptive mother was one of Prince Fushimi Kuniie's daughters, but her biological mother was Tamiko Niihata, the daughter of a doctor from the Ichijō family. Unusually for the time, she had been vaccinated against smallpox. As a child, Masako was somewhat of a prodigy: she was able to read poetry from the Kokin Wakashū by the age of 4 and had composed some waka verses of her own by the age of 5. By age seven, she was able to read some texts in classical Chinese with some assistance and was studying Japanese calligraphy. By the age of 12, she had studied the koto and was fond of Noh drama. She excelled in the studies of finances, ikebana and Japanese tea ceremony.The major obstacle to Lady Masako's eligibility to become empress consort was the fact that she was 3 years older than Emperor Meiji, but this issue was resolved by changing her official birth date from 1849 to 1850. They became engaged on 2 September 1867, when she adopted the given name Haruko (美子), which was intended to reflect her \nserene beauty and diminutive size.\nThe Tokugawa Bakufu promised 15,000 ryō in gold for the wedding and assigned her an annual income of 500 koku, but as the Meiji Restoration occurred before the wedding could be completed, the promised amounts were never delivered. The wedding was delayed partly due to periods of mourning for Emperor Kōmei,  for her brother Saneyoshi, and the political disturbances around Kyoto between 1867 and 1868.\n\nEmpress of Japan\nLady Haruko and Emperor Meiji's wedding was finally officially celebrated on 11 January 1869. She was the first imperial consort to receive the title of both nyōgō and of kōgō (literally, the emperor's wife, translated as \"empress consort\"), in several hundred years. However, it soon became clear that she was unable to bear children. Emperor Meiji already had 12 children by 5 concubines, though: as custom in Japanese monarchy, Empress Haruko adopted Yoshihito, her husband's eldest son by Lady Yanagihara Naruko, who became Crown Prince. On 8 November 1869, the Imperial House departed from Kyoto for the new capital of Tokyo. In a break from tradition, Emperor Meiji insisted that the Empress and the senior ladies-in-waiting should attend the educational lectures given to the Emperor on a regular basis about national conditions and developments in foreign nations.\n\nInfluence\nOn 30 July 1886, Empress Haruko attended the Peeresses School's graduation ceremony in Western clothing. On 10 August, the imperial couple received foreign guests in Western clothing for the first time when hosting a Western Music concert.From this point onward, the Empress' entourage wore only Western-style clothes in public, to the point that in January 1887 \nEmpress Haruko issued a memorandum on the subject: traditional Japanese dress was not only unsuited to modern life, but Western-style dress was closer than the kimono to clothes worn by Japanese women in ancient times.In the diplomatic field, Empress Haruko hosted the wife of former US President Ulysses S. Grant during his visit to Japan. She was also present for her husband's meetings with Hawaiian King Kalākaua in 1881. Later that same year, she helped host the visit of the sons of future British King Edward VII: Princes Albert Victor and George (future George V), who presented her with a pair of pet wallabies from Australia.On  26 November 1886, Empress Haruko accompanied her husband to Yokosuka, Kanagawa to observe the new Imperial Japanese Navy cruisers Naniwa and Takachiho firing torpedoes and performing other maneuvers. From 1887, the Empress was often at the Emperor's side in official visits to army maneuvers. When Emperor Meiji fell ill in 1888, Empress Haruko took his place in welcoming envoys from Siam, launching warships and visiting Tokyo Imperial University. In 1889, Empress Haruko accompanied Emperor Meiji on his official visit to Nagoya and Kyoto. While he continued on to visit naval bases at Kure and Sasebo, she went to Nara to worship at the principal Shinto shrines.Known throughout her tenure for her support of charity work and women's education during the First Sino-Japanese War (1894–95), Empress Haruko worked for the establishment of the Japanese Red Cross Society. She participated in the organization's administration, especially in their peacetime activities in which she created a money fund for the International Red Cross. Renamed \"The Empress Shōken Fund\", it is presently used for international welfare activities. After Emperor Meiji moved his military headquarters from Tokyo to Hiroshima to be closer to the lines of communications with his troops, Empress Haruko joined her husband in March 1895. While in Hiroshima, she insisted on visiting hospitals full of wounded soldiers every other day of her stay.\n\nDeath\nAfter Emperor Meiji's death in 1912, Empress Haruko was granted the title Empress Dowager (皇太后, Kōtaigō) by her adoptive son, Emperor Taishō. She died in 1914 at the Imperial Villa in Numazu, Shizuoka and was buried in the East Mound of the Fushimi Momoyama Ryo in Fushimi, Kyoto, next to her husband. Her soul was enshrined in Meiji Shrine in Tokyo. On 9 May 1914, she received the posthumous name Shōken Kōtaigō (昭憲皇太后). Her railway-carriage can be seen today in the Meiji Mura Museum, in Inuyama, Aichi prefecture.\n\nHonours\nNational\nGrand Cordon of the Order of the Precious Crown, 1 November 1888\n\nForeign\nShe received the following orders and decorations:\n Russian Empire: Grand Cross of the Order of St. Catherine, 13 December 1887\n Spain: Dame of the Order of Queen Maria Luisa, 29 November 1889\n Siam: Dame of the Order of the Royal House of Chakri, 12 October 1899\n German Empire: Dame of the Order of Louise, 1st Class, 19 May 1903\n Kingdom of Bavaria: Dame of Honour of the Order of Theresa, 29 February 1904\n Korean Empire: Grand Cordon of the Order of the Auspicious Phoenix, 27 July 1908\n\nAncestry\nSee also\nEmpress of Japan\nŌmiya Palace\n\nNotes\nPassage 6:\nEunoë (wife of Bogudes)\nEunoë Maura was the wife of Bogudes, King of Western Mauretania. Her name has also been spelled Euries or Euryes or Eunoa.\n\nBiography\nEarly life\nEunoë Maura was thought to be descended from Berbers, but her name is Greek so it appears she might have been from there or had Greek ancestry. She was likely of very high status, as she is mentioned by historian Suetonius in the same context as Cleopatra.\n\nMarriage\nAt an unspecified early date in her marriage to her husband Bogud he mounted an expedition along the Atlantic coast, seemingly venturing into the tropics. When he returned he presented his wife Eunoë with gigantic reeds and asparagus he had found on the journey.She is believed to have been a mistress of Julius Caesar. She may have replaced Cleopatra in Caesar's affections, when he arrived in North Africa prior to the Battle of Thapsus on 6 April 46 BC, the two were among several queens courted by Caesar. It is also possible that they first met in Spain if she accompanied her husband there on a campaign. Only a brief romance for the Roman, both Eunoe and Bogudes profited through gifts bestowed on them by Caesar. Caesar departed from Africa in June 46 BC, five and a half months after he landed.\n\nCultural depictions\nEunoë and Caesar's affair is greatly exaggerated and expanded on in the Medieval French prose work Faits des Romains. Jeanette Beer in her book A Medieval Caesar states that the Roman general is \"transformed into Caesar, the medieval chevalier\" in the text, and that the author is more interested in Caesar's sexual dominance over the queen than the political dominance he held over her husband Bogud. The text describes her; \"Eunoe was the most beautiful woman in four kingdoms — nevertheless, she was Moorish\", which Beer further analysed as being indicative of the fact that it was unimaginable to audiences of the time to believe that a lover of Caesar could be ugly, but that Moors still represented everything that was ugly to them.Eunoë has also been depicted in several novels about Caesar, as well as serialized stories in The Cornhill Magazine. In such fiction her character often serves as a foil for the relationship between Caesar and another woman, mostly Cleopatra, such as in The Memoirs of Cleopatra, The Bloodied Toga and When We Were Gods. In Song of the Nile she also plays a posthumous role as a person of interest for Cleopatra's daughter Selene II who became queen of Mauritania after her.Eunoe has also been depicted in a numismatic drawing by Italian artist and polymath Jacopo Strada, who lived in the 16th century. There is however no archaeological evidence of a coin that bears her name or picture.\n\nSee also\nWomen in ancient Rome\nPassage 7:\nCatherine Exley\nCatherine Exley (1779–1857) was an English diarist. She was the wife of a soldier who accompanied her husband when he served in Portugal, Spain, and Ireland during the Napoleonic Wars. Exley is best known as the author of a diary that gives an account of military life in that era from the viewpoint of the wife of a common soldier.\n\nBackground\nCatherine Whitaker was born at Leeds in 1779 and married Joshua Exley there in 1806. Between  1805 and 1815, Joshua served in the Second Battalion of the 34th Regiment of Foot, initially as a private and then for a little over two years, as a corporal. Exley accompanied her husband for a substantial portion of this time and in due course wrote an account that is probably unique in that it records and reflects on life in the British Army from the perspective of the wife of a soldier who did not reach the rank of an officer.\n\nThe diary\nCatherine's diary was first published as a booklet issued shortly after her death. A single copy of the booklet is known to exist, it was also reprinted in The Dewsbury Reporter during August 1923. The text of the diary is included in full in a more recently issued book, edited by Professor Rebecca Probert, along with essays on its military and religious context, the treatment of prisoners of war and the role of women in the British, French and Spanish armed forces during the Peninsular War.\nThe diary unfolds the hardships that both Catherine and her husband suffered during his military service, including one period when they both wrongly thought that the other had died. There are detailed accounts of the births and deaths of children, the cold, hunger and filthy conditions of military life and the horror of the aftermaths of battles. Details of the author's religious experiences which led her to membership of the Methodist church also appear. Exley wrote the diary during the last 20 years before her death, which took place in 1857 at Batley, Yorkshire.\nPassage 8:\nIlona Zrínyi\nCountess Ilona Zrínyi (Croatian: Jelena Zrinska, Hungarian: Zrínyi Ilona) (1643, Ozalj – 18 February 1703, Izmit) was a noblewoman and heroine. She was one of the last surviving members of the Croatian-Hungarian Zrinski/Zrínyi noble family. She was the daughter of Petar Zrinski, Ban (viceroy) of Croatia, the niece of both Miklós Zrínyi and Fran Krsto Frankopan and the wife of Francis Rákóczi I and Imre Thököly, as well as the mother of Francis Rákóczi II. She is remembered in history for her Defense of Palanok Castle against the Imperial army in 1685-1688, an act for which she was regarded a heroine in Hungary.\n\nLife\nEarly years and family\nIlona was born Ilona Zrínyi in Ozalj, present day Croatia. She was the eldest child of Croatian Ban, Peter Zrinyi, and his wife Katarina Zrinyi née Frankopan, a Croatian poet. Later her parents had two daughters, Judita Petronila (1652-1699), and Aurora Veronika (1658-1735), as well as a son, Ivan Antun (1651-1703). Ilona and her siblings were the last generation of descendants of the once-powerful Zrinski family.\nFrom her childhood, she was known for her beauty and good education. There is little information on her schooling; it is known though that she acquired a high level of knowledge within her family, not only from her father and mother, Croatian writers and erudite persons but from her uncle Nikola VII Zrinski as well.\n\nMarriages\nOn 1 March 1666, she married Francis Rákóczi, with whom she had three children: György, born in 1667, who died in infancy; Julianna, born in 1672; and Ferenc (commonly known as Francis Rákóczi II), born in 1676. On June 8, 1676, not long after Francis II's birth, the elder Francis died. The widowed Ilona requested guardianship of her children and was granted it, against the advice of Emperor Leopold I's advisers and against Francis I's will. In this way she also retained control over the vast Rákóczi estates, which included among them the castles of Regéc, Sárospatak, Makovica, and Munkács. In 1682 she married Imre Thököly and became an active partner in her second husband's Kuruc uprising against the Habsburgs.\n\nDefense of Munkács (Palanok) Castle\nAfter their defeat at the 1683 Battle of Vienna, both the Ottoman forces and Thököly's allied Kuruc fighters had no choice but to retreat, and Thököly quickly lost one Rákóczi castle after another. At the end of 1685, the Imperial army surrounded the last remaining stronghold, Munkacs Castle in today's Ukraine. Ilona Zrínyi alone defended the castle for three years (1685–1688) against the forces of General Antonio Caraffa.\n\nInternment, exile and death\nAfter the recapture of Buda, the situation became untenable, and on 17 January 1688, Ilona had no choice but to surrender the castle, with the understanding that the defenders would receive amnesty from the Emperor, and that the Rákóczi estates would remain in her children's name. Under this agreement, she and her children traveled immediately to Vienna, where in violation of the pact the children were taken from her. Ilona lived until 1691 in the convent of the Ursulines, where her daughter Julianna was also raised. Her son Francis was immediately taken to the Jesuit school in Neuhaus.\nAt the time, her husband, Thököly, was still fighting with his Kuruc rebels against the Habsburg army in Upper Hungary. When Habsburg General Heisler was captured by Thököly, a prisoner exchange was arranged, and Ilona joined her husband in Transylvania. In 1699, however, after the Treaty of Karlowitz was signed, both spouses, having found themselves on the losing side, had to go into exile in the Ottoman Empire. The countess lived in Galata, district of Constantinople, and later in Izmit, where she died on 18 February 1703. She was buried in the French church of Saint Benoit in Galata.\n\nDescendants\nFrom her first marriage with Francis Rákóczi, Ilona had three children:\n\nGyörgy (1667–1667)\nJulianna Borbála (September 1672 – 1717); married Count Ferdinand Gobert von Aspremont-Lynden (1643-1708)\nFrancis II (27 March 1676 – 8 April 1735)From her second marriage with Imre Thököly, Ilona had three children, all of whom died at a young age (including one she was pregnant with during the siege of Munkács).\n\nLegacy\nIlona Zrínyi is celebrated in Croatia and Hungary as one of the greatest national heroines, patriots and fighters for freedom, who opposed, although unsuccessfully, the autocracy and absolutism aspirations of the Habsburgs. Her even more famous son Francis II Rákóczi continued the struggle for the independence of Hungary (1703–1711).\nIn October 1906 the remains of the Croatian countess were reinterred with her son's in the St Elisabeth Cathedral in present-day Košice, Slovakia.\n\nHonors\nPostage stamp issued by Hungary on 28 September 1952.\n\nSee also\nHouse of Zrinski\nZrinski family tree\nZrinski–Frankopan conspiracy\nKuruc\nRákóczi's War for Independence\nWesselényi conspiracy\nPassage 9:\nArtaynte\nArtaynte (f. 478 BC), was the wife of the Crown Prince Darius.\n\nLife\nDaughter of an unnamed woman and Prince Masistes, a marshall of the armies during the invasion of Greece in 480-479 BC, and the brother of King Xerxes I.\nDuring the Greek campaign Xerxes developed a passionate desire for the wife of Masistes, but she would constantly resist and would not bend to his will. Upon his return to Sardis, the king endeavoured to bring about the marriage of his son Daris to Artaynte, the daughter of this woman the wife of Masistes, supposing that by doing so he could obtain her more easily.\nAfter moving to Susa he brought Artaynte to the royal house with him for his son Daris, but fell in love with her himself, and after obtaining her they became lovers. \nAt the behest of Xerxes, Artaynte committed adultery with him (Xerxes). When queen Amestris found out, she did not seek revenge against Artaynte, but against her mother, Masistes' wife, as Amestris thought that it was her connivance. On Xerxes' birthday, Amestris sent for his guards and mutilated Masistes' wife by cutting off her breasts and threw them to dogs, and her nose and ears and lips also, and cutting out her tongue as well. On seeing this, Masistes fled to Bactria to start a revolt, but was intercepted by Xerxes' army who killed him and his sons.\nPassage 10:\nHafsa Hatun\nHafsa Hatun (Ottoman Turkish: حفصه خاتون, \"young lioness\") was a Turkish princess, and a consort of Bayezid I, Sultan of the Ottoman Empire.\n\nLife\nHafsa Hatun was the daughter of Isa Bey, the ruler of the Aydinids. She was married to Bayezid in 1390, upon his conquest of the Aydinids. Her father had surrendered without a fight, and a marriage was arranged between her and Bayezid. Thereafter, Isa was sent into exile in Iznik, shorn of his power, where he subsequently died. Her marriage strengthened the bonds between the two families.\n\nCharities\nHafsa Hatun's public works are located within her father's territory and may have been built before she married Bayezid. She commissioned a fountain in Tire city and a Hermitage in Bademiye, and a mosque known as \"Hafsa Hatun Mosque\" between 1390 and 1392 from the money she received in her dowry.\n\nSee also\nOttoman dynasty\nOttoman Empire",
                       "answers": ["Ozalj"], "dataset": "2wikimqa"}

        systemPrompt = """
            You are a helpful assistant.
            Please read the following Passages and answer the Question below.
        """
        passages = re.findall(r"Passage\s+(\d+):(.*?)(?=Passage\s+\d+:|$)", dataset_row['context'], re.S)
        chunks = [f"Passage {i}:{passages[i][1]}" for i in range(len(passages))]
        # 构造问答请求
        question = (
                '\n Question: '
                + dataset_row['input']
                + 'Answer within 5 words.'
        )
        origin_sys_prompt_ids = tokenizer.encode(systemPrompt)
        padded_sys_prompt_ids = pad_rag_chunks(origin_sys_prompt_ids, block_size, chunk_pad_token_id, chunk_end_token_id)
        # 1. sys prompt warm up
        print(f"---------------1. sys prompt: warm up---------------")
        get_output(llm, TokensPrompt(prompt_token_ids=padded_sys_prompt_ids), prefill_sampling_params)
        time.sleep(0.5)

        padded_contexts_ids = []
        origin_prompt_ids = origin_sys_prompt_ids
        padded_prompt_ids = padded_sys_prompt_ids
        for text_chunk in chunks:
            un_pad_ids = tokenizer.encode(text_chunk, add_special_tokens=False)
            padded_ids = pad_rag_chunks(un_pad_ids, block_size, chunk_pad_token_id, chunk_end_token_id)
            padded_prompt_ids = padded_prompt_ids + padded_ids
            origin_prompt_ids = origin_prompt_ids + un_pad_ids
            padded_contexts_ids.append(padded_ids)

        question_ids = tokenizer.encode(question, add_special_tokens=False)
        padded_prompt_ids = padded_prompt_ids + question_ids
        origin_prompt_ids = origin_prompt_ids + question_ids

        print(f"--------------- baseline with no cache blend ---------------")
        # for nfs store,you should clean up DATA_DIR(/home/data/kv_cache) to get the full prefill compute time
        baseline_time , baseline_gen_text = get_output(llm, TokensPrompt(prompt_token_ids=padded_prompt_ids), sampling_params)
        time.sleep(0.5)

        print(f"--------------- cache rag chunks ---------------")
        llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in padded_contexts_ids], sampling_params)
        time.sleep(0.5)

        print(f"--------------- warm up blend code ---------------")
        warm_up_blend_prompt_ids = padded_sys_prompt_ids
        for ids in reversed(padded_contexts_ids):
            warm_up_blend_prompt_ids = warm_up_blend_prompt_ids + ids
        warm_up_blend_prompt_ids = warm_up_blend_prompt_ids + question_ids
        llm.generate(TokensPrompt(prompt_token_ids=warm_up_blend_prompt_ids), sampling_params)
        time.sleep(0.5)

        print(f"--------------- rag chunks cache blend ---------------")
        blend_time , blend_gen_text = get_output(llm, TokensPrompt(prompt_token_ids=padded_prompt_ids), sampling_params)

        print(f"Baseline generated text: {baseline_gen_text!r}")
        print(f"Baseline generated cost time: {baseline_time:.2f} seconds")
        print(f"Blend generated text: {blend_gen_text!r}")
        print(f"Blend generated cost time: {blend_time:.2f} seconds")
        print(f"Question:{dataset_row['input']}")
        print(f"Golden answer:{dataset_row["answers"]}")

if __name__ == "__main__":
    main()
