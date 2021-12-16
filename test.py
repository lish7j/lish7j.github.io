from torch import nn, optim
import pandas as pd

from graph.model import GAT

from utils import generate_adj
from utils import cosSim

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = 'cuda'

EMBEDDING_DIM = 100
HIDDEN_DIM = 128
MAX_LENGTH = 40
EPOCHS = 40
ATTRIBUTE_DIM = 128



class KnowledgeBase():
    def __init__(self, data_path):
        self.diseases, self.symptoms, self.aliases,  self.parts, self.drugs, \
        self.disease_to_alias, self.disease_to_symptom, self.disease_to_part,\
                self.disease_to_drug = self.read_file(data_path)

        self.entity_dict, self.entity_num = self.__generate_entity_dict()
        self.id_to_entity = {k:v for v, k in self.entity_dict.items()}
        self.edges = self.__generate_edges()
        self.node_nums = len(self.entity_dict)
        
        print(self.node_nums, self.entity_num)
        self.idx = torch.arange(0, self.node_nums)
        
    def read_file(self, data_path):
        """
        读取文件，获得实体，实体关系
        """
        # cols = ["name", "alias", "part", "age", "infection", "insurance", "department", "checklist", "symptom",
        #         "complication", "treatment", "drug", "period", "rate", "money"]
        # 实体
        diseases = []  # 疾病
        aliases = []  # 别名
        symptoms = []  # 症状
        parts = []  # 部位
        drugs = []  # 药品

        # 疾病的属性：age, infection, insurance, checklist, treatment, period, rate, money
    
        # 关系
        disease_to_symptom = []  # 疾病与症状关系
        disease_to_alias = []  # 疾病与别名关系
        diseases_to_part = []  # 疾病与部位关系
        disease_to_drug = []  # 疾病与药品关系

        all_data = pd.read_csv(data_path, encoding='gb18030').loc[:, :].values
        for i, data in enumerate(all_data):
            if i == 300:
                break
            disease_dict = {}  # 疾病信息
            # 疾病
            disease = str(data[0]).replace("...", " ").strip()
            diseases.append(disease)
            disease_dict["name"] = disease
            # 别名
            line = re.sub("[，、；,.;]", " ", str(data[1])) if str(data[1]) else "未知"
            for alias in line.strip().split():
                aliases.append(alias)
                disease_to_alias.append([disease, alias])
            # 部位
            part_list = str(data[2]).strip().split() if str(data[2]) else "未知"
            for part in part_list:
                parts.append(part)
                diseases_to_part.append([disease, part])
        
            
            
            # 症状
            symptom_list = str(data[8]).replace("...", " ").strip().split()[:-1]
            for symptom in symptom_list:
                symptoms.append(symptom)
                disease_to_symptom.append([disease, symptom])
            
            
            # 药品
            drug_string = str(data[11]).replace("...", " ").strip()
            for drug in drug_string.split()[:-1]:
                drugs.append(drug)
                disease_to_drug.append([disease, drug])
        

        return set(diseases), set(symptoms), set(aliases), set(parts), \
                set(drugs), disease_to_alias, disease_to_symptom, diseases_to_part,\
                disease_to_drug, 
    
    def __generate_entity_dict(self):
        """
            给每个实体分配一个id
        """
        entity_dict = dict()
        id = 0
        for d in self.diseases:
            entity_dict[d+'_disease'] = id
            id += 1
        for s in self.symptoms:
            entity_dict[s+'_symptom'] = id
            id += 1
        
        for a in self.aliases:
            entity_dict[a+'_aliase'] = id
            id += 1
        for p in self.parts:
            entity_dict[p+'_part'] = id
            id += 1
        for d in self.drugs:
            entity_dict[d+'_drug'] = id
            id += 1
        
        return entity_dict, id
        
    def __generate_edges(self):
        """
            根据给定的实体边和实体字典，生成边集,如（1，2）
        """
        edges = list()
        for da in self.disease_to_alias:
            d, a = da[0]+'_disease', da[1]+'_aliase'
            edges.append([self.entity_dict[d], self.entity_dict[a]])
        for ds in self.disease_to_symptom:
            d, s = ds[0]+'_disease', ds[1]+'_symptom'
            edges.append([self.entity_dict[d], self.entity_dict[s]])
        for dp in self.disease_to_part:
            d, p = dp[0]+'_disease', dp[1]+'_part'
            edges.append([self.entity_dict[d], self.entity_dict[p]])
        for dd in self.disease_to_drug:
            d1, d2 = dd[0]+'_disease', dd[1]+'_drug'
            edges.append([self.entity_dict[d1], self.entity_dict[d2]])
        return edges

    def get_adj(self):
        edges = np.array(self.edges)
        return generate_adj(edges, self.node_nums).clone().detach().requires_grad_(False)

    def get_pair(self, entity):
        matched = self.entity_dict[entity]
        while True:
            unmtched = random.randint(0, self.node_nums - 1)
            if matched != unmtched:
                return [matched, unmtched]

    def construct_train_graph(self, data):
        
        aliases = set()    # 疾病别名
        parts = set()      # 疾病部位
        symptoms = set()   # 症状
        drugs = set()      # 需要药品信息

        # 疾病
        disease = str(data[0]).replace("...", " ").strip() + '_disease'
        
        # 别名
        line = re.sub("[，、；,.;]", " ", str(data[1])) if str(data[1]) else "未知"
        for alias in line.strip().split():
            aliases.add(alias+'_aliase')
            
        # 部位
        part_list = str(data[2]).strip().split() if str(data[2]) else "未知"
        for part in part_list:
            parts.add(part+'_part')
            
        # 症状
        symptom_list = str(data[8]).replace("...", " ").strip().split()[:-1]
        for symptom in symptom_list:
            symptoms.add(symptom+'_symptom')

        # 药品
        drug_string = str(data[11]).replace("...", " ").strip()
        for drug in drug_string.split()[:-1]:
            drugs.add(drug+'_drug')

        entity_to_id = dict()
        entity_to_id[disease] = 0
    
        entitys = aliases.union(symptoms).union(drugs).union(parts)
        idx = 1
        for entity in entitys:
            entity_to_id[entity] = idx
            idx += 1
        id_to_entity = {k:v for v, k in entity_to_id.items()}

        node_nums = len(entitys) + 1
        edges = []
        for i in range(1, node_nums):
            edges.append([0, i])
            edges.append([i, 0])
        edges = np.array(edges)
        adj = generate_adj(edges, node_nums).clone().detach().requires_grad_(False)
        
        train_data = []
        

        idx = []
        idx.append(self.entity_dict[entity])
        for entity in entitys:
            id = self.entity_dict[entity]
            idx.append(id)
            id_ = entity_to_id[entity]
            pair = self.get_pair(entity)
            train_data.append([id_, pair[0], pair[1]])
        
        
        return adj, torch.tensor(idx), entitys, entity_to_id

    def construct_test_graph(self, data):
    
        aliases = set()    # 疾病别名
        parts = set()      # 疾病部位
        symptoms = set()   # 症状
        drugs = set()      # 需要药品信息

        # 疾病
        disease = str(data[0]).replace("...", " ").strip() + '_disease'
        
        # 别名
        line = re.sub("[，、；,.;]", " ", str(data[1])) if str(data[1]) else "未知"
        for alias in line.strip().split():
            aliases.add(alias+'_aliase')
            
        # 部位
        part_list = str(data[2]).strip().split() if str(data[2]) else "未知"
        for part in part_list:
            parts.add(part+'_part')
            
        # 症状
        symptom_list = str(data[8]).replace("...", " ").strip().split()[:-1]
        for symptom in symptom_list:
            symptoms.add(symptom+'_symptom')

        # 药品
        drug_string = str(data[11]).replace("...", " ").strip()
        for drug in drug_string.split()[:-1]:
            drugs.add(drug+'_drug')

        entity_to_id = dict()
        entity_to_id[disease] = 0
    
        entitys = aliases.union(symptoms).union(drugs).union(parts)
        idx = 1
        for entity in entitys:
            entity_to_id[entity] = idx
            idx += 1
        id_to_entity = {k:v for v, k in entity_to_id.items()}

        node_nums = len(entitys) + 1
        edges = []
        for i in range(1, node_nums):
            edges.append([0, i])
            edges.append([i, 0])
        edges = np.array(edges)
        adj = generate_adj(edges, node_nums).clone().detach().requires_grad_(False)
        
        pos_train_data = []
        
        idx = []
        idx.append(self.entity_dict[entity])
        for entity in entitys:
            id = self.entity_dict[entity]
            idx.append(id)
            id_ = entity_to_id[entity]
            pos_train_data.append([id_, id])
            
        return adj, torch.tensor(idx), entitys, entity_to_id


class EDGNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, device):
        super(EDGNN, self).__init__()
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.model = GAT(128, 128, 128, 0.5, 0.2, 2).to(device)
        
    
    def forward(self, idx, adj):
        features = self.word_embeds(idx).to(device)
        return self.model(features, adj)


def cos(x, y):
    x_sum = torch.sqrt(torch.sum(torch.square(x), dim=-1, dtype=torch.float32))
    y_sum = torch.sqrt(torch.sum(torch.square(y), dim=-1, dtype=torch.float32))
    return torch.dot(x, y) / (x_sum * y_sum + 1)


if __name__ == "__main__":
    log_path = './test_data/new_real_test.txt'
    kb = KnowledgeBase("/home/lishaoji/my_project/data/disease.csv")
    model = EDGNN(kb.node_nums, 128, device)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = 0.0015, weight_decay=0.001)
    loss_func = nn.LogSigmoid().to(device)

    
    all_data = pd.read_csv('/home/lishaoji/my_project/data/disease.csv', encoding='gb18030').loc[:, :].values
    kb_id2entity = kb.id_to_entity

    train_lines = []
    test_lines = []
    for lines, data in enumerate(all_data):
        if lines == 280:
            break
        if lines < 250:    
            train_lines.append(data)   
        else: 
            test_lines.append(data)

    print("train {0} test {1}".format(len(train_lines), len(test_lines)))
    
    train_data_set = []
    test_data_set  = []
    for data in train_lines:
        adj, idx, entitys, entity_to_id = kb.construct_train_graph(data) 
        train_data_set.append([adj, idx, entitys, entity_to_id])
    for data in test_lines:
        adj, idx, entitys, entity_to_id = kb.construct_test_graph(data)
        test_data_set.append([adj, idx, entitys, entity_to_id])

    kb_idx = kb.idx.to(device)
    kb_adj = kb.get_adj().to(device)

    for epoch in range(EPOCHS):
        model.train()
        count = 0
        loss_all = 0.0
        for adj, idx, entitys, entity_to_id in train_data_set:
            if len(entitys) == 0:
                continue
            adj = adj.to(device)
            idx = idx.to(device)
            loss_total = torch.tensor(0, dtype=torch.float32, requires_grad=False).to(device)
                
            h1 = model(idx, adj)
            h2 = model(kb_idx, kb_adj)

            for entity in entitys:
                id1 = entity_to_id[entity]
                id2, id3 = kb.get_pair(entity)        
                x = h1[id1].to(device)
                y, z = h2[id2].to(device), h2[id3].to(device)
                print(id1, id2, entity)
                loss = loss_func(cos(x, y))
                loss_total = loss_total - loss
                print(loss, loss_total)
                #loss_total = loss_total - (x, z)

            count += 1
            loss_all += loss_total.item()
            loss_total.backward()
            optimizer.step()
            optimizer.zero_grad()

        correct = 0
        all = 0
        print("== epochs{0}, loss={1} ==", loss_all / count)

        model.eval()
        with torch.no_grad(): 
            for adj, idx, entitys, entity_to_id in test_data_set:
                adj = adj.to(device)
                idx = idx.to(device)

                h1 = model(idx, adj)
                h2 = model(kb_idx, kb_adj)
                
                for entity in entitys:
                    id1 = entity_to_id[entity]
                    id2, id3 = kb.get_pair(entity)        
                    h_x = h1[id1].to(device)
                    h_true = h2[id2].to(device)
                    y_pred = 0
                    sim = -1
                    candidates = set()
                    for node in range(kb.node_nums):
                        e2 = kb_id2entity[node]
                        if str(entity.split('_')[1]) != str(e2.split('_')[1]):
                            continue
                        h_y = h2[node]
                        sim_2 = cos(h_x, h_y)
                        candidates.add(sim_2.item())

                    all += 1
                    sim_true = cos(h_x, h_true).item()
                    sim_list = list(candidates)
                    sim_list.sort()
                    most_sim = set(sim_list[-10:])
        
                    if most_sim.__contains__(sim_true):
                        correct += 1
                        
        print("epochs={0}, test acc={1:.5f},all={2}".format(epoch, correct/all, all))   
        # torch.save(model.state_dict(), "./models_chinese/nn-head=2_gat_epochs={0}_testAcc={1:.6f}".format(epoch, correct/all))
