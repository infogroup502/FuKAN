from model.RevIN import RevIN
import time
from utils.utils import *
from model.FuKAN import FuKAN
from data_factory.data_loader import get_loader_segment
from metrics.metrics import *
import warnings
warnings.filterwarnings('ignore')


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class Solver(object):
    DEFAULTS = {}
    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.train_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
                                               win_size=self.win_size, mode='train', dataset=self.dataset, )
        self.vali_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
                                              win_size=self.win_size, mode='val', dataset=self.dataset)
        self.test_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
                                              win_size=self.win_size, mode='test', dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
                                              win_size=self.win_size, mode='thre', dataset=self.dataset)
        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.loss_fuc == 'MAE':
            self.criterion = nn.L1Loss()
        elif self.loss_fuc == 'MSE':
            self.criterion = nn.MSELoss()
            self.criterion_keep= nn.MSELoss(reduction='none')


    def build_model(self):
        self.model = FuKAN(win_size=self.win_size, d_model=self.d_model, local_size=self.local_size, global_size=self.global_size, channel=self.input_c,seq_len=self.seq_len)
        if torch.cuda.is_available():
            self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self):
        time_now = time.time()
        train_steps = len(self.train_loader)  # 3866
        for epoch in range(self.num_epochs):
            iter_count = 0
            epoch_time = time.time()
            self.model.train()
            for it, (input_data, labels) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)  # (128,100,51)
                revin_layer = RevIN(num_features=self.input_c)
                x = revin_layer(input, 'norm')
                B, L, M = x.shape
                x_local_front = []
                x_global_front = []
                x_local_back = []
                x_global_back = []
                for index, localsize in enumerate(self.local_size):
                    num = self.seq_len*localsize#local_front
                    boundary = self.win_size-num
                    result = []
                    result1 = []
                    for i in range(self.win_size):
                        if(i<num-1):
                            temp = x[:,0,:].unsqueeze(1).repeat(1, num - i-1, 1)
                            temp1 = torch.cat((temp,x[:,0:i+1,:]),dim=1)
                            result.append(temp1)
                        else:
                            result.append(x[:,i-num+1:i+1,:])

                    for i in range(self.win_size):
                        if (i > boundary):
                            temp = x[:, self.win_size - 1, :].unsqueeze(1).repeat(1, num + i - self.win_size, 1)
                            temp1 = torch.cat((x[:, i:self.win_size, :], temp), dim=1)
                            result1.append(temp1)
                        else:
                            result1.append(x[:, i:i + num, :])
                    local_front = torch.cat(result,axis=0).reshape(L, B, num, M).permute(1, 0, 3, 2)
                    local_back = torch.cat(result1,axis=0).reshape(L, B, num, M).permute(1, 0, 3, 2)


                    result = []
                    result1 = []
                    base_sequence = np.arange(self.global_size[0])
                    segment_offsets = np.arange(self.seq_len)
                    for i in range(self.win_size):
                        start_offsets = base_sequence * num
                        start_positions = start_offsets + i
                        indices = start_positions[:, np.newaxis] + segment_offsets
                        flat_indices = indices.flatten()
                        flat_indices[flat_indices >= self.win_size] = self.win_size - 1
                        result.append(x[:,flat_indices,:])
                        start_positions = i - start_offsets
                        indices = start_positions[:, np.newaxis] - segment_offsets
                        flat_indices = indices.flatten()
                        flat_indices[flat_indices <0] = 0
                        result1.append(x[:, flat_indices, :])
                    global_front = torch.cat(result1,axis=0).reshape(L, B, self.seq_len*self.global_size[0], M).permute(1, 0, 3, 2)
                    global_back  = torch.cat(result,axis=0).reshape(L, B, self.seq_len*self.global_size[0], M).permute(1, 0, 3, 2)

                    x_local_front.append(local_front)
                    x_global_front.append(global_front)
                    x_local_back.append(local_back)
                    x_global_back.append(global_back)

                local_front, global_front, local_fuzzy_front, global_fuzzy_front,local_back, global_back, local_fuzzy_back, global_fuzzy_back = self.model(x_local_front, x_global_front,x_local_back,x_global_back)

                local_loss_front = 0.0
                global_loss_front = 0.0
                local_loss_back = 0.0
                global_loss_back = 0.0
                contr_loss_front = 0.0
                contr_loss_back = 0.0
                for u in range(len(local_front)):
                    local_loss_front += self.criterion(local_front[u],global_fuzzy_front[u])
                    global_loss_front += self.criterion(global_front[u],local_fuzzy_front[u])
                    local_loss_back += self.criterion(local_back[u], global_fuzzy_back[u])
                    global_loss_back += self.criterion(global_back[u], local_fuzzy_back[u])

                loss = contr_loss_front+local_loss_front+global_loss_front+contr_loss_back+local_loss_back+global_loss_back
                if (it + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    if speed > 1.5:
                        return 1
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - it)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                loss.backward(retain_graph=True)
                self.optimizer.step()
            print(
                "Epoch: {0}, Cost time: {1:.3f}s ".format(
                    epoch + 1, time.time() - epoch_time))
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def test(self):
        op = "contr"
        kk=3
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            revin_layer = RevIN(num_features=self.input_c)
            x = revin_layer(input, 'norm')
            B, L, M = x.shape
            x_local_front = []
            x_global_front = []
            x_local_back = []
            x_global_back = []
            for index, localsize in enumerate(self.local_size):
                num = self.seq_len * localsize  # local_front
                boundary = self.win_size - num
                result = []
                result1 = []
                for i in range(self.win_size):
                    if (i < num - 1):
                        temp = x[:, 0, :].unsqueeze(1).repeat(1, num - i - 1, 1)
                        temp1 = torch.cat((temp, x[:, 0:i + 1, :]), dim=1)
                        result.append(temp1)
                    else:
                        result.append(x[:, i - num + 1:i + 1, :])

                local_front = torch.cat(result, axis=0).reshape(L, B, num, M).permute(1, 0, 3, 2)
                local_back = torch.flip(local_front, [-1])

                result = []
                result1 = []
                base_sequence = np.arange(self.global_size[0])
                segment_offsets = np.arange(self.seq_len)
                for i in range(self.win_size):
                    start_offsets = base_sequence * num
                    start_positions = i - start_offsets
                    indices = start_positions[:, np.newaxis] - segment_offsets
                    flat_indices = indices.flatten()
                    flat_indices[flat_indices < 0] = 0
                    result1.append(x[:, flat_indices, :])
                global_front = torch.cat(result1, axis=0).reshape(L, B, self.seq_len * self.global_size[0], M).permute(
                    1, 0, 3, 2)
                global_back = torch.flip(global_front, [-1])
                x_local_front.append(local_front)
                x_global_front.append(global_front)
                x_local_back.append(local_back)
                x_global_back.append(global_back)

            local_front, global_front, local_fuzzy_front, global_fuzzy_front, local_back, global_back, local_fuzzy_back, global_fuzzy_back= self.model(
                x_local_front, x_global_front, x_local_back, x_global_back)

            local_loss_front = 0.0
            global_loss_front = 0.0
            local_loss_back = 0.0
            global_loss_back = 0.0
            contr_loss_front = 0.0
            contr_loss_back = 0.0
            for u in range(len(local_front)):
                    local_loss_front += torch.sum(self.criterion_keep(local_front[u], global_fuzzy_front[u]),dim=-1)
                    global_loss_front += torch.sum(self.criterion_keep(global_front[u],local_fuzzy_front[u]),dim=-1)
                    local_loss_back += torch.sum(self.criterion_keep(local_back[u], global_fuzzy_back[u]), dim=-1)
                    global_loss_back += torch.sum(self.criterion_keep(global_back[u], local_fuzzy_back[u]), dim=-1)
            local_loss_front, _ = torch.topk(local_loss_front, k=kk, dim=-1)
            global_loss_front, _ = torch.topk(global_loss_front, k=kk, dim=-1)
            local_loss_back, _ = torch.topk(local_loss_back, k=kk, dim=-1)
            global_loss_back, _ = torch.topk(global_loss_back, k=kk, dim=-1)

            local_loss_front = torch.mean(local_loss_front, dim=-1)
            global_loss_front = torch.mean(global_loss_front, dim=-1)
            local_loss_back = torch.mean(local_loss_back, dim=-1)
            global_loss_back = torch.mean(global_loss_back, dim=-1)
            score_front = local_loss_front+global_loss_front
            score_back = local_loss_back+global_loss_back
            score,_ = torch.max(torch.cat((score_front.unsqueeze(-1),score_back.unsqueeze(-1)),dim=-1),dim=-1)
            metric = torch.softmax(score, dim=-1)

            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)

            revin_layer = RevIN(num_features=self.input_c)
            x = revin_layer(input, 'norm')
            B, L, M = x.shape
            x_local_front = []
            x_global_front = []
            x_local_back = []
            x_global_back = []
            for index, localsize in enumerate(self.local_size):
                num = self.seq_len * localsize  # local_front
                boundary = self.win_size - num
                result = []
                result1 = []
                for i in range(self.win_size):
                    if (i < num - 1):
                        temp = x[:, 0, :].unsqueeze(1).repeat(1, num - i - 1, 1)
                        temp1 = torch.cat((temp, x[:, 0:i + 1, :]), dim=1)
                        result.append(temp1)
                    else:
                        result.append(x[:, i - num + 1:i + 1, :])

                local_front = torch.cat(result, axis=0).reshape(L, B, num, M).permute(1, 0, 3, 2)
                local_back = torch.flip(local_front, [-1])
                result = []
                result1 = []
                base_sequence = np.arange(self.global_size[0])
                segment_offsets = np.arange(self.seq_len)
                for i in range(self.win_size):
                    start_offsets = base_sequence * num
                    start_positions = i - start_offsets
                    indices = start_positions[:, np.newaxis] - segment_offsets
                    flat_indices = indices.flatten()
                    flat_indices[flat_indices < 0] = 0
                    result1.append(x[:, flat_indices, :])
                global_front = torch.cat(result1, axis=0).reshape(L, B, self.seq_len * self.global_size[0], M).permute(
                    1, 0, 3, 2)
                global_back = torch.flip(global_front, [-1])

                x_local_front.append(local_front)
                x_global_front.append(global_front)
                x_local_back.append(local_back)
                x_global_back.append(global_back)

            local_front, global_front, local_fuzzy_front, global_fuzzy_front, local_back, global_back, local_fuzzy_back, global_fuzzy_back= self.model(
                x_local_front, x_global_front, x_local_back, x_global_back)

            local_loss_front = 0.0
            global_loss_front = 0.0
            local_loss_back = 0.0
            global_loss_back = 0.0
            contr_loss_front = 0.0
            contr_loss_back = 0.0
            for u in range(len(local_front)):
                local_loss_front += torch.sum(self.criterion_keep(local_front[u], global_fuzzy_front[u]), dim=-1)
                global_loss_front += torch.sum(self.criterion_keep(global_front[u], local_fuzzy_front[u]), dim=-1)
                local_loss_back += torch.sum(self.criterion_keep(local_back[u], global_fuzzy_back[u]), dim=-1)
                global_loss_back += torch.sum(self.criterion_keep(global_back[u], local_fuzzy_back[u]), dim=-1)
            local_loss_front, _ = torch.topk(local_loss_front, kk, dim=-1)
            global_loss_front, _ = torch.topk(global_loss_front, k=kk, dim=-1)
            local_loss_back, _ = torch.topk(local_loss_back, k=kk, dim=-1)
            global_loss_back, _ = torch.topk(global_loss_back, k=kk, dim=-1)

            local_loss_front = torch.mean(local_loss_front, dim=-1)
            global_loss_front = torch.mean(global_loss_front, dim=-1)
            local_loss_back = torch.mean(local_loss_back, dim=-1)
            global_loss_back = torch.mean(global_loss_back, dim=-1)
            score_front = local_loss_front+global_loss_front
            score_back = local_loss_back+global_loss_back
            score,_ = torch.max(torch.cat((score_front.unsqueeze(-1),score_back.unsqueeze(-1)),dim=-1),dim=-1)
            metric = torch.softmax(score, dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("anormly_ratio",self.anormly_ratio)
        print("Threshold :", thresh)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)

            revin_layer = RevIN(num_features=self.input_c)
            x = revin_layer(input, 'norm')
            B, L, M = x.shape
            x_local_front = []
            x_global_front = []
            x_local_back = []
            x_global_back = []
            for index, localsize in enumerate(self.local_size):
                num = self.seq_len * localsize  # local_front
                boundary = self.win_size - num
                result = []
                result1 = []
                for i in range(self.win_size):
                    if (i < num - 1):
                        temp = x[:, 0, :].unsqueeze(1).repeat(1, num - i - 1, 1)
                        temp1 = torch.cat((temp, x[:, 0:i + 1, :]), dim=1)
                        result.append(temp1)
                    else:
                        result.append(x[:, i - num + 1:i + 1, :])

                local_front = torch.cat(result, axis=0).reshape(L, B, num, M).permute(1, 0, 3, 2)
                local_back = torch.flip(local_front, [-1])
                result = []
                result1 = []
                base_sequence = np.arange(self.global_size[0])
                segment_offsets = np.arange(self.seq_len)
                for i in range(self.win_size):
                    start_offsets = base_sequence * num
                    start_positions = i - start_offsets
                    indices = start_positions[:, np.newaxis] - segment_offsets
                    flat_indices = indices.flatten()
                    flat_indices[flat_indices < 0] = 0

                    result1.append(x[:, flat_indices, :])
                global_front = torch.cat(result1, axis=0).reshape(L, B, self.seq_len * self.global_size[0], M).permute(
                    1, 0, 3, 2)
                global_back = torch.flip(global_front, [-1])
                x_local_front.append(local_front)
                x_global_front.append(global_front)
                x_local_back.append(local_back)
                x_global_back.append(global_back)

            local_front, global_front, local_fuzzy_front, global_fuzzy_front, local_back, global_back, local_fuzzy_back, global_fuzzy_back = self.model(
                x_local_front, x_global_front, x_local_back, x_global_back)

            local_loss_front = 0.0
            global_loss_front = 0.0
            local_loss_back = 0.0
            global_loss_back = 0.0
            contr_loss_front = 0.0
            contr_loss_back = 0.0
            for u in range(len(local_front)):
                local_loss_front += torch.sum(self.criterion_keep(local_front[u], global_fuzzy_front[u]), dim=-1)
                global_loss_front += torch.sum(self.criterion_keep(global_front[u], local_fuzzy_front[u]), dim=-1)
                local_loss_back += torch.sum(self.criterion_keep(local_back[u], global_fuzzy_back[u]), dim=-1)
                global_loss_back += torch.sum(self.criterion_keep(global_back[u], local_fuzzy_back[u]), dim=-1)
            local_loss_front, _ = torch.topk(local_loss_front, k=kk, dim=-1)
            global_loss_front, _ = torch.topk(global_loss_front, k=kk, dim=-1)
            local_loss_back, _ = torch.topk(local_loss_back, k=kk, dim=-1)
            global_loss_back, _ = torch.topk(global_loss_back, k=kk, dim=-1)

            local_loss_front = torch.mean(local_loss_front, dim=-1)
            global_loss_front = torch.mean(global_loss_front, dim=-1)
            local_loss_back = torch.mean(local_loss_back, dim=-1)
            global_loss_back = torch.mean(global_loss_back, dim=-1)
            score_front = local_loss_front+global_loss_front
            score_back = local_loss_back+global_loss_back
            score,_ = torch.max(torch.cat((score_front.unsqueeze(-1),score_back.unsqueeze(-1)),dim=-1),dim=-1)
            metric = torch.softmax(score, dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        pred = (test_energy >= thresh).astype(int)
        gt = test_labels.astype(int)
        from sklearn.metrics import f1_score
        micro = f1_score(gt, pred, average='micro')
        print("micro", micro)
        print("macro", f1_score(gt, pred, average='macro'))
        print("weighted", f1_score(gt, pred, average='weighted'))
        print("binary", f1_score(gt, pred, average='binary'))


        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score
        from metrics.combine_all_scores import combine_all_evaluation_scores
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')

        matrix = [self.index]
        scores_simple = combine_all_evaluation_scores(pred, gt, test_energy)
        for key, value in scores_simple.items():
            matrix.append(value)
            print('{0:21} : {1:0.4f}'.format(key, value))

        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision,
                                                                                                   recall, f_score))

        if self.data_path == 'UCR' or 'UCR_AUG':
            import csv
            with open('result/' + self.data_path + '.csv', 'a+') as f:
                writer = csv.writer(f)
                writer.writerow(matrix)

        return accuracy, precision, recall, f_score