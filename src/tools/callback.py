
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""callback function"""
import os

from mindspore.train.callback import Callback
from mindspore import save_checkpoint

from src.args import args


class EvaluateCallBack(Callback):
    """EvaluateCallBack"""

    def __init__(self, model, eval_dataset, src_url, train_url):
        super(EvaluateCallBack, self).__init__()
        self.model = model
        self.eval_dataset = eval_dataset
        self.train_url = train_url
        self.best_acc = 0.

        self.src_url = "/cache/best"
        os.makedirs(self.src_url, exist_ok = True)

    def epoch_end(self, run_context):
        """
            Test when epoch end, save best model with best.ckpt.
        """
        
        cb_params = run_context.original_args()
        cur_epoch_num = cb_params.cur_epoch_num
        train_network = cb_params.train_network

        if cur_epoch_num >= 270 or cur_epoch_num == 10:
            result = self.model.eval(self.eval_dataset)

            if result["acc"] > self.best_acc:
                self.best_acc = result["acc"]

                best_epoch = os.path.join(self.src_url, f'epoch_{cur_epoch_num:02d}_{self.best_acc:.3f}_best.ckpt')
                save_checkpoint(train_network, best_epoch)

                print("epoch: %s acc: %s, best acc is %s" % (cb_params.cur_epoch_num, result["acc"], self.best_acc), flush=True)
                
                if args.run_modelarts:
                    import moxing as mox
                    mox.file.copy_parallel(src_url = self.src_url, dst_url = self.train_url)
