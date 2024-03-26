# -*- coding: utf-8 -*-
# %%
from sfm.data.sci_data.SFMDecTokenizer import SFMDecTokenizer
from tqdm import tqdm
# %%
tokenizer = SFMDecTokenizer.from_pretrained(
    '/hai1/ds_dataset/llama2/llama-2-7b',
    prot_spm_path='/blob/shufxi/data/scigpt/ur50bpe/bpe',
    dna_spm_path='/blob/shufxi/data/scigpt/dnabpe/bpe'
)

# %%
tokenizer.tokenize('<fragA>CCC</fragA>')
# %%
tokenizer.convert_tokens_to_ids(tokenizer.tokenize('<protein>EAAAAAAAAA</protein>'))
# %%
def tokenize(line):
    print(line)
    tokens = tokenizer.tokenize(line)
    print(tokens)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(ids)

# %%
tokenize('<protein>A')
# %%
tokenize('<protein>PSETLSLTCAVYGGSFSGYYWSWIRQPPGKGLEWIGEINHSGSTNYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCAKVLSEGYLPGYYNPFDSWGRGTLVTVSS</protein>')
# %%
tokenize('<protein>MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSEKGLIVGHFSGIKYKGEKAQASEVDVNKMCCWVSKFKDAMRRYQGIQTCKIPGKVLSDLDAKIKAYNLTVEGVEGFVRYSRVTKQHVAAFLKELRHSKQYENVNLIHYILTDKRVDIQHLEKDLVKDFKALVESAHRMRQGHMINVKYILYQLLKKHGHGPDGPDILTVKTGSKGVLYDDSFRKIYTDLGWKFTPL</protein>')
# %%
text="""
Bovine liver dihydrofolate reductase: purification and properties of the enzyme. A purification procedure is reported for obtaining bovine liver dihydrofolate reductase in high yield and amounts of 100-200 mg. A key step in the procedure is the use of an affinity gel prepared by coupling pteroyl-L-lysine to Sepharose. The purified reductase has a specific activity of about 100 units/mg and is homogeneous as judged by analytical ultracentrifugation, polyacrylamide gel electrophoresis, and titration with methotrexate. The products of the first step of Edman degradation indicated a minimum purity of 79%. The reductase has a molecular weight of about 21500 on the basis of amino acid composition and 22100 +/- 300 from equilibrium sedimentation. It is not inhibited by antiserum to the <protein>MAAPATVALAVKAAITAATDKRTRNAVCILVAALVTPLILIIVMIVSLLSAAADHNNTAIDLCFNGGAISSQAPADYAAHIRDMRGSFSELDAAITDISAELEDGSLDSIRIKAIFYALLFGAENLRMDNSGYQAFVKCFVGYETRTRTVDHGDGTTSEETRTVAVPIKSLPEIYDNLENTLGRTITLEDQANAAEIYYRILYGGSVPTYGKAFDQWANGLPLSDAPFVGADGFCSPLGENWRGVVTSEFGYRTDPFTGESRGHTGLDLGAPSGTPIRAALDGTVQFVRYTNTGYGYHLAIDHGGGFVTLYGHCSKILIAEGQTVKAGDIIAQVGSTGRSTGPHLHFEVRINGEMKNPRSYLP</protein> (isoenzyme 2). Unlike the reductase of many other vertebrate tissues, the bovine enzyme is inhibited by mercurials rather than activated and it has a single pH optimum at both low and high ionic strength. However, the position of the pH optimum is shifted and the activity increased by increasing ionic strength. Automatic Edman degradation has been used to determine 34 of the amino-terminal 37 amino acid residues. Considerable homology exists between this region and the corresponding regions of the reductase from S. faecium and from Escherichia coli. This strengthens the idea that this region contributes to the structure of the binding site for dihydrofolate.
"""

tokenize(text)
# %%
tokenize('<mol>c1cc([*:100001])cc([*:100002])c1.c1ccc2c(c1)c1ccccc1n2[*:100000]</mol>')
# %%
path = "/blob/yinxia/wu2/shared/SFM/SFM.overall.data/wrapped_data/PMC_v1_wrapped.txt"
lines = open(path).readlines()




# %%
new_tokens = set()
for line in tqdm(lines):
    tokens = tokenizer.tokenize(line.strip())
    ids = tokenizer.convert_tokens_to_ids(tokens)
    for tok, idx in zip(tokens, ids):
        if idx == 0:
            new_tokens.add(tok)
    if len(new_tokens) > 10:
        print(tokens)
        print(ids)
        break

print(new_tokens)
# %%
