# Cell実験用 測定をでぽられーとにしてみた
import sys
import netsquid as ns
import pandas
from netsquid.components.qprocessor import QuantumProcessor, PhysicalInstruction
from netsquid.nodes import Node, Connection, Network
from netsquid.protocols.protocol import Signals
from netsquid.protocols.nodeprotocols import NodeProtocol
from netsquid.components.qchannel import QuantumChannel
from netsquid.components.cchannel import ClassicalChannel
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.qubits.state_sampler import StateSampler
from netsquid.components.qprogram import QuantumProgram
from netsquid.components.models.qerrormodels import FibreLossModel, DepolarNoiseModel, DephaseNoiseModel
from netsquid.components.models.delaymodels import FibreDelayModel, FixedDelayModel
from netsquid.util.datacollector import DataCollector
import pydynaa
from netsquid.qubits import ketstates as ks
from netsquid.qubits import qubitapi as qapi
from netsquid.components import instructions as instr
from netsquid.components.instructions import Instruction
from netsquid.qubits.operators import Operator, I, X, Y
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# モジュールのインポートパスを確認
print(sys.path)

__all__ = [
    "ExternalEntanglingConnection",
    "ClassicalConnection",
    "InitStateProgram",
    "BellMeasurementProgram",
    "BellMeasurementProtocol",
    "CorrectionProtocol",
    "create_processor",
    "create_client_processor",
    "example_network_setup",
    "example_sim_setup",
    "run_dis_experiment",
    "create_dis_plot",
]
# グローバルフラグの初期値
# def global_set():
#     global global_finished_bell
#     global global_finished_correction
#     global_finished_bell = False
#     global_finished_correction = False
global_finished_bell = False
global_finished_correction = False
#
# 1) メモリが空いているかどうかをチェックする関数
#
def memory_is_available(mem_position):
    """メモリポジションに qubit が載っていない (is_empty=True) かどうかだけを確認"""
    # print(f"メモリ位置 {mem_position} の空き状況: {mem_position.is_empty}")
    return mem_position.is_empty

#
# 2) EXTERNALモードで動くQSourceを持つためのConnectionクラス(改変後)
#
class ExternalEntanglingConnection(Connection):
    """
    QSourceを内蔵しないで、量子チャネルだけ持つ Connection。
    ただしサブコンポーネントとして QSource を追加する。
    """
    def __init__(self, length, name="ExternalEntanglingConnection", fidelity=1.0):
        super().__init__(name=name)

        # 量子チャネルの設定
        qchannel_c2a = QuantumChannel(
            "qchannel_C2A",
            length=length / 2,
            models={
                "quantum_loss_model": FibreLossModel(p_loss_init=0, p_loss_length=0),
                "quantum_noise_model": DepolarNoiseModel(depolar_rate=0)
            }
        )
        qchannel_c2b = QuantumChannel(
            "qchannel_C2B",
            length=length / 2,
            models={
                "quantum_loss_model": FibreLossModel(p_loss_init=0, p_loss_length=0),
                "quantum_noise_model": DepolarNoiseModel(depolar_rate=0)
            }
        )
        # サブコンポーネントに追加
        self.add_subcomponent(qchannel_c2a, forward_output=[("A", "recv")])
        self.add_subcomponent(qchannel_c2b, forward_output=[("B", "recv")])

        # fidelityを指定してQSourceを作成
        qsource = create_external_qsource(name="AliceQSource", fidelity=fidelity)
        self.add_subcomponent(qsource)

        # QSourceのポートをQuantumChannelの送信ポートに接続
        qsource.ports["qout0"].connect(qchannel_c2a.ports["send"])
        qsource.ports["qout1"].connect(qchannel_c2b.ports["send"])

#
# 3) QSourceを段階的に待機してトリガーするプロトコル　アリスとボブの両方を考慮
#
class ExternalSourceProtocol(NodeProtocol):
    """
    - 最初に base_delay [ns] 待つ
    - アリスとボブのメモリが空いていなければ extra_delay [ns] を足して再度待機
    - 両方のメモリが空き次第 qsource.trigger() する
    """

    def __init__(self, node, qsource, other_node, mem_pos_a=3, mem_pos_b=0,
                 base_delay=1e9/200, extra_delay=1e6, max_retries=3000000):
        """
        Parameters
        ----------
        node : Node
            プロトコルを動かす側のノード (アリス)
        qsource : QSource
            EXTERNAL モードで生成した QSource
        other_node : Node
            メモリ状態をチェックする他のノード (ボブ)
        mem_pos_a : int
            アリスのメモリポジション番号
        mem_pos_b : int
            ボブのメモリポジション番号
        base_delay : float
            最初に待つ時間 [ns]
        extra_delay : float
            メモリが空いていなかった場合に追加で待機する時間 [ns]
        max_retries : int
            何度まで再待機を繰り返すか
        """
        super().__init__(node)
        self.qsource = qsource
        self.other_node = other_node
        self.mem_pos_a = mem_pos_a
        self.mem_pos_b = mem_pos_b
        self.base_delay = base_delay
        self.extra_delay = extra_delay
        self.max_retries = max_retries

    def run(self):
        # まず base_delay 待つ
        while True:

            yield self.await_timer(self.base_delay)

            retries = 0
            while True:
                # アリスのメモリポジションを確認
                mem_position_a = self.node.qmemory.mem_positions[self.mem_pos_a]
                # ボブのメモリポジションを確認
                mem_position_b = self.other_node.qmemory.mem_positions[self.mem_pos_b]

                if memory_is_available(mem_position_a) and memory_is_available(mem_position_b):
                    # 両方のメモリが空いていれば、QSourceをトリガーしてペア生成
                    # print(f"[{ns.sim_time()} ns] アリスとボブのメモリが空いたので qsource.trigger() します。")
                    self.qsource.trigger()
                    break
                else:
                    if retries >= self.max_retries:
                        # print(f"[{ns.sim_time()} ns] メモリ空き待ちリトライ上限（max_retries={self.max_retries}）に達しました。中断します。")
                        break
                    # まだ空いていない → extra_delay 待って再チェック
                    # print(f"[{ns.sim_time()} ns] メモリが埋まっているので {self.extra_delay} ns 待つ (retries={retries})")
                    retries += 1
                    yield self.await_timer(self.extra_delay)

            # print(f"[{ns.sim_time()} ns] ExternalSourceProtocol 終了")

#
# 4) ClassicalConnection クラス (変更なし)
#
class ClassicalConnection(Connection):
    """A connection that transmits classical messages in one direction, from A to B."""
    def __init__(self, length, name="ClassicalConnection"):
        super().__init__(name=name)
        self.add_subcomponent(
            ClassicalChannel("Channel_A2B", length=length,
                             models={"delay_model": FibreDelayModel(c=200000)}),
            forward_input=[("A", "send")],
            forward_output=[("B", "recv")]
        )


from netsquid.components.models.qerrormodels import DepolarNoiseModel
from netsquid.components.qprocessor import PhysicalInstruction
from netsquid.components import instructions as instr
from netsquid.components.models.qerrormodels import T1T2NoiseModel

def create_processor(dephase_rate=0.0039,T1=1e10,T2_ratio=0.1,sge=None,dge=None,gate_speed_factor=1.0, gate_durations=None):
    """ゲート速度を可変にした量子プロセッサを作成するファクトリ関数"""
    if gate_durations is None:
        # デフォルトのゲート実行時間 [ns]
        gate_durations = {
            instr.INSTR_INIT: 1000,
            instr.INSTR_H: 135000,
            instr.INSTR_X: 135000,
            instr.INSTR_Z: 135000,
            instr.INSTR_Y: 135000,
            instr.INSTR_S: 135000,
            instr.INSTR_ROT_X: 135000,
            instr.INSTR_ROT_Y: 135000,
            instr.INSTR_ROT_Z: 135000,
            instr.INSTR_CNOT: 600000,
            instr.INSTR_MEASURE: 200000
        }
    
    # ゲート速度ファクターを適用
    scaled_gate_durations = {k: v / gate_speed_factor for k, v in gate_durations.items()}
    
    # メモリノイズモデル
    memory_noise_model = T1T2NoiseModel(T1=T1, T2=T1*T2_ratio)
    
    
    # ゲートごとのノイズモデルを設定
    gate_noise_models = {
        instr.INSTR_H: DepolarNoiseModel(depolar_rate=sge,time_independent=True),  # Xゲートのフィデリティ99%
        instr.INSTR_X: DepolarNoiseModel(depolar_rate=sge,time_independent=True), # Hゲートのフィデリティ99.5%
        instr.INSTR_Z: DepolarNoiseModel(depolar_rate=sge,time_independent=True),
        instr.INSTR_Y: DepolarNoiseModel(depolar_rate=sge,time_independent=True),
        instr.INSTR_ROT_X: DepolarNoiseModel(depolar_rate=sge,time_independent=True),
        instr.INSTR_ROT_Z: DepolarNoiseModel(depolar_rate=sge,time_independent=True),
        instr.INSTR_ROT_Y: DepolarNoiseModel(depolar_rate=sge,time_independent=True),
        instr.INSTR_CNOT: DepolarNoiseModel(depolar_rate=dge,time_independent=True),
        # 他のゲートも同様に設定可能
    }
    
    # PhysicalInstructionのリストを作成
    physical_instructions = [
        PhysicalInstruction(instr.INSTR_INIT, duration=scaled_gate_durations[instr.INSTR_INIT], parallel=True),
        PhysicalInstruction(instr.INSTR_H, duration=scaled_gate_durations[instr.INSTR_H], parallel=True, topology=[0, 1, 2, 3],
                           quantum_noise_model=gate_noise_models.get(instr.INSTR_H, None)),
        PhysicalInstruction(instr.INSTR_X, duration=scaled_gate_durations[instr.INSTR_X], parallel=True, topology=[0, 1, 2, 3],
                           quantum_noise_model=gate_noise_models.get(instr.INSTR_X, None)),
        PhysicalInstruction(instr.INSTR_Z, duration=scaled_gate_durations[instr.INSTR_Z], parallel=True, topology=[0, 1, 2, 3]),
        PhysicalInstruction(instr.INSTR_Y, duration=scaled_gate_durations[instr.INSTR_Y], parallel=True, topology=[0, 1, 2, 3]),
        PhysicalInstruction(instr.INSTR_S, duration=scaled_gate_durations[instr.INSTR_S], parallel=True, topology=[0, 1, 2, 3]),
        PhysicalInstruction(instr.INSTR_ROT_X, duration=scaled_gate_durations[instr.INSTR_ROT_X], parallel=True, topology=[0, 1, 2, 3]),
        PhysicalInstruction(instr.INSTR_ROT_Y, duration=scaled_gate_durations[instr.INSTR_ROT_Y], parallel=True, topology=[0, 1, 2, 3]),
        PhysicalInstruction(instr.INSTR_ROT_Z, duration=scaled_gate_durations[instr.INSTR_ROT_Z], parallel=True, topology=[0, 1, 2, 3]),
        PhysicalInstruction(instr.INSTR_CNOT, duration=scaled_gate_durations[instr.INSTR_CNOT], parallel=True, topology=[(0, 1)]),
        PhysicalInstruction(instr.INSTR_CNOT, duration=scaled_gate_durations[instr.INSTR_CNOT], parallel=True, topology=[(0, 2)]),
        PhysicalInstruction(instr.INSTR_CNOT, duration=scaled_gate_durations[instr.INSTR_CNOT], parallel=True, topology=[(1, 2)]),
        PhysicalInstruction(instr.INSTR_CNOT, duration=scaled_gate_durations[instr.INSTR_CNOT], parallel=True, topology=[(2, 3)]),
        PhysicalInstruction(instr.INSTR_MEASURE, duration=scaled_gate_durations[instr.INSTR_MEASURE], parallel=False, topology=[0],
                           quantum_noise_model=DepolarNoiseModel(depolar_rate=dephase_rate, time_independent=True),
                           apply_q_noise_after=False),
        PhysicalInstruction(instr.INSTR_MEASURE, duration=scaled_gate_durations[instr.INSTR_MEASURE], parallel=False, topology=[1],
                           quantum_noise_model=DepolarNoiseModel(depolar_rate=dephase_rate, time_independent=True),
                           apply_q_noise_after=False),
        PhysicalInstruction(instr.INSTR_MEASURE, duration=scaled_gate_durations[instr.INSTR_MEASURE], parallel=False, topology=[2],
                           quantum_noise_model=DepolarNoiseModel(depolar_rate=dephase_rate, time_independent=True),
                           apply_q_noise_after=False),
        PhysicalInstruction(instr.INSTR_MEASURE, duration=scaled_gate_durations[instr.INSTR_MEASURE], parallel=False, topology=[3],
                           quantum_noise_model=DepolarNoiseModel(depolar_rate=dephase_rate, time_independent=True),
                           apply_q_noise_after=False)
    ]
    
    # 量子プロセッサの作成
    processor = QuantumProcessor("quantum_processor", num_positions=4,
                                 memory_noise_models=[memory_noise_model] * 4,
                                 phys_instructions=physical_instructions)
    return processor

# def create_client_processor(dephase_rate=0.0039,T1=1e10,T2_ratio=0.1,sge=None,dge=None,gate_speed_factor=1.0, gate_durations=None):
#     """ゲート速度を可変にした量子プロセッサを作成するファクトリ関数"""
#     if gate_durations is None:
#         # デフォルトのゲート実行時間 [ns]
#         gate_durations = {
#             instr.INSTR_INIT: 1000,
#             instr.INSTR_H: 135000,
#             instr.INSTR_X: 135000,
#             instr.INSTR_Z: 135000,
#             instr.INSTR_Y: 135000,
#             instr.INSTR_S: 135000,
#             instr.INSTR_ROT_X: 135000,
#             instr.INSTR_ROT_Y: 135000,
#             instr.INSTR_ROT_Z: 135000,
#             instr.INSTR_CNOT: 600000,
#             instr.INSTR_MEASURE: 200000
#         }
    
#     # ゲート速度ファクターを適用
#     scaled_gate_durations = {k: v / gate_speed_factor for k, v in gate_durations.items()}
    
#     # メモリノイズモデル
#     memory_noise_model = T1T2NoiseModel(T1=T1, T2=T1*T2_ratio)
#     gate_noise_models = {
#         instr.INSTR_H: DepolarNoiseModel(depolar_rate=sge,time_independent=True),  # Xゲートのフィデリティ99%
#         instr.INSTR_X: DepolarNoiseModel(depolar_rate=sge,time_independent=True), # Hゲートのフィデリティ99.5%
#         instr.INSTR_Z: DepolarNoiseModel(depolar_rate=sge,time_independent=True),
#         instr.INSTR_Y: DepolarNoiseModel(depolar_rate=sge,time_independent=True),
#         instr.INSTR_ROT_X: DepolarNoiseModel(depolar_rate=sge,time_independent=True),
#         instr.INSTR_ROT_Z: DepolarNoiseModel(depolar_rate=sge,time_independent=True),
#         instr.INSTR_ROT_Y: DepolarNoiseModel(depolar_rate=sge,time_independent=True),
#         instr.INSTR_CNOT: DepolarNoiseModel(depolar_rate=dge,time_independent=True),
#         # 他のゲートも同様に設定可能
#     }
    
#     # PhysicalInstructionのリストを作成
#     physical_instructions = [
#         PhysicalInstruction(instr.INSTR_ROT_Z, duration=scaled_gate_durations[instr.INSTR_ROT_Z], parallel=True, topology=[0]),
#         PhysicalInstruction(instr.INSTR_MEASURE, duration=scaled_gate_durations[instr.INSTR_MEASURE], parallel=False, topology=[0],
#                            quantum_noise_model=DepolarNoiseModel(depolar_rate=dephase_rate, time_independent=True),
#                            apply_q_noise_after=False)
                           
#     ]
    
#     # 量子プロセッサの作成
#     processor = QuantumProcessor("quantum_processor", num_positions=1,
#                                  memory_noise_models=[memory_noise_model] * 1,
#                                  phys_instructions=physical_instructions)
#     return processor

def create_client_processor(dephase_rate=0.0039, T1=1e10, T2_ratio=0.1, sge=None, dge=None, gate_speed_factor=1.0, gate_durations=None):
    """クライアント用の量子プロセッサを作成するファクトリ関数"""
    if gate_durations is None:
        # デフォルトのゲート実行時間 [ns]
        gate_durations = {
            instr.INSTR_INIT: 1000,
            instr.INSTR_H: 135000,
            instr.INSTR_X: 135000,
            instr.INSTR_Z: 135000,
            instr.INSTR_Y: 135000,
            instr.INSTR_S: 135000,
            instr.INSTR_ROT_X: 135000,
            instr.INSTR_ROT_Y: 135000,
            instr.INSTR_ROT_Z: 135000,
            instr.INSTR_CNOT: 600000,
            instr.INSTR_MEASURE: 200000
        }
    
    # ゲート速度ファクターを適用
    scaled_gate_durations = {k: v / gate_speed_factor for k, v in gate_durations.items()}
    
    # メモリノイズモデル
    memory_noise_model = T1T2NoiseModel(T1=T1, T2=T1*T2_ratio)
    
    # ゲートノイズモデル
    gate_noise_models = {
        instr.INSTR_H: DepolarNoiseModel(depolar_rate=sge, time_independent=True),
        instr.INSTR_X: DepolarNoiseModel(depolar_rate=sge, time_independent=True),
        instr.INSTR_Z: DepolarNoiseModel(depolar_rate=sge, time_independent=True),
        instr.INSTR_Y: DepolarNoiseModel(depolar_rate=sge, time_independent=True),
        instr.INSTR_ROT_X: DepolarNoiseModel(depolar_rate=sge, time_independent=True),
        instr.INSTR_ROT_Z: DepolarNoiseModel(depolar_rate=sge, time_independent=True),
        instr.INSTR_ROT_Y: DepolarNoiseModel(depolar_rate=sge, time_independent=True)
    }
    
    # PhysicalInstructionのリスト - クライアントプログラムが必要とする命令を全て含める
    physical_instructions = [
        # ClientProgramが使用するH命令を追加
        PhysicalInstruction(instr.INSTR_H, duration=scaled_gate_durations[instr.INSTR_H], 
                          parallel=True, topology=[0],
                          quantum_noise_model=gate_noise_models.get(instr.INSTR_H)),
        
        # ROT_Z命令は必須
        PhysicalInstruction(instr.INSTR_ROT_Z, duration=scaled_gate_durations[instr.INSTR_ROT_Z], 
                          parallel=True, topology=[0],
                          quantum_noise_model=gate_noise_models.get(instr.INSTR_ROT_Z)),
        
        # 念のため他の単一量子ビット命令も追加
        PhysicalInstruction(instr.INSTR_X, duration=scaled_gate_durations[instr.INSTR_X], 
                          parallel=True, topology=[0],
                          quantum_noise_model=gate_noise_models.get(instr.INSTR_X)),
        
        PhysicalInstruction(instr.INSTR_Z, duration=scaled_gate_durations[instr.INSTR_Z], 
                          parallel=True, topology=[0],
                          quantum_noise_model=gate_noise_models.get(instr.INSTR_Z)),
        
        PhysicalInstruction(instr.INSTR_Y, duration=scaled_gate_durations[instr.INSTR_Y], 
                          parallel=True, topology=[0],
                          quantum_noise_model=gate_noise_models.get(instr.INSTR_Y)),
        
        # 測定命令
        PhysicalInstruction(instr.INSTR_MEASURE, duration=scaled_gate_durations[instr.INSTR_MEASURE], 
                          parallel=False, topology=[0],
                          quantum_noise_model=DepolarNoiseModel(depolar_rate=dephase_rate, time_independent=True),
                          apply_q_noise_after=False)
    ]
    
    # 量子プロセッサの作成 - クライアントは1量子ビットのみ
    processor = QuantumProcessor("client_quantum_processor", num_positions=1,
                                memory_noise_models=[memory_noise_model],
                                phys_instructions=physical_instructions)
    return processor


#
# 6) もつれ生成用の QSource を EXTERNAL で作るファクトリ
#
def create_external_qsource(name="ExtQSource", fidelity=1.0):
    """EXTERNALモードのQSourceを作成するファクトリ関数。
    fidelity < 1.0 の場合、生成されるベル対に Depolarizing Noise を適用し、忠実度を下げる。
    """
    from netsquid.qubits import ketstates as ks
    
    # Fidelity F を depolarizing の確率 p に変換
    p_depol = 4/3 * (1 - fidelity)
    if p_depol < 0:
        p_depol = 0.0  # fidelityが1.0を超えたりした場合の保護（無音で切り捨てる）
    
    qsource = QSource(
        name=name,
        state_sampler=StateSampler([ks.b00], [1.0]),
        num_ports=2,
        status=SourceStatus.EXTERNAL,
        models={
            # 生成した瞬間に Depolarizing チャネルを通過したとみなす
            "emission_noise_model": DepolarNoiseModel(
                depolar_rate=p_depol,
                time_independent=True  # 時間に依存しないノイズ
            )
        }
    )
    return qsource

#
# 7) 既存プログラム類 (InitStateProgram, BellMeasurementProgram 等) は変更なし
#
class InitStateProgram(QuantumProgram):
    default_num_qubits = 3
    def program(self):
        q1, q2, q3 = self.get_qubit_indices(3)
        self.apply(instr.INSTR_INIT, q1)
        self.apply(instr.INSTR_INIT, q2)
        self.apply(instr.INSTR_INIT, q3)
        self.apply(instr.INSTR_ROT_Y, q1, angle=np.pi / 2)
        self.apply(instr.INSTR_X, q2)
        self.apply(instr.INSTR_ROT_X, q2, angle=-np.pi / 2)
        self.apply(instr.INSTR_CNOT, [q1, q2])
        self.apply(instr.INSTR_CNOT, [q2, q3])
        self.apply(instr.INSTR_CNOT, [q1, q2])
        self.apply(instr.INSTR_ROT_Y, q1, angle=-np.pi / 2)
        self.apply(instr.INSTR_ROT_X, q2, angle=np.pi / 2)
        yield self.run()


class BellMeasurementProgram(QuantumProgram):
    """2量子ビットに対するベル測定を行うプログラム"""
    default_num_qubits = 4
    def program(self):
        q1, q2, q3, q4 = self.get_qubit_indices(4)
        self.apply(instr.INSTR_CNOT, [q3, q4])
        self.apply(instr.INSTR_H, q3)
        self.apply(instr.INSTR_MEASURE, q3, output_key="M3")
        self.apply(instr.INSTR_MEASURE, q4, output_key="M4")
        yield self.run()


class ClientProgram(QuantumProgram):
    """1量子ビットを測定するだけの簡単プログラム"""
    default_num_qubits = 1
    def program(self):
        # q1 = self.get_qubit_indices(1)[0]
        q1 = self.get_qubit_indices(1)
        self.apply(instr.INSTR_H, q1)
        self.apply(instr.INSTR_MEASURE, q1, output_key="M5")
        yield self.run()


class ServerProgram(QuantumProgram):
    """2量子ビットを測定するだけの簡単プログラム"""
    default_num_qubits = 2
    def program(self):
        q1, q2 = self.get_qubit_indices(2)
        self.apply(instr.INSTR_MEASURE, q1, output_key="M1")
        self.apply(instr.INSTR_MEASURE, q2, output_key="M2")
        yield self.run()


class M5XServerProgram(QuantumProgram):
    default_num_qubits = 2
    def program(self):
        q1, q2 = self.get_qubit_indices(2)
        self.apply(instr.INSTR_X, q2)
        self.apply(instr.INSTR_MEASURE, q1, output_key="M1")
        self.apply(instr.INSTR_MEASURE, q2, output_key="M2")
        yield self.run()


class Collect1Program(QuantumProgram):
    default_num_qubits = 3
    def program(self):
        q1, q2, q3 = self.get_qubit_indices(3)
        self.apply(instr.INSTR_CNOT, [q1, q3])
        self.apply(instr.INSTR_H, q1)
        yield self.run()


class Collect2Program(QuantumProgram):
    default_num_qubits = 3
    def program(self):
        q1, q2, q3 = self.get_qubit_indices(3)
        self.apply(instr.INSTR_CNOT, [q2, q3])
        self.apply(instr.INSTR_H, q2)
        yield self.run()

class XmeasureProgram(QuantumProgram):
    default_num_qubits = 2
    def program(self):
        q1, q2 = self.get_qubit_indices(2)
        self.apply(instr.INSTR_H, q1)
        self.apply(instr.INSTR_H, q2)
        yield self.run()

#
# 8) BellMeasurementProtocol, CorrectionProtocol 等
#
def memory_status(node):
    """ノードのメモリ状態を表示"""
    for pos in node.qmemory.mem_positions:
        print(f"Position {pos} is_empty: {pos.is_empty}")

class BellMeasurementProtocol(NodeProtocol):
    
    def __init__(self, node, shot_times_list, band_times_list, server_times_list, serverlist, max_runs,flag):
        super().__init__(node)
        self.shot_times_list = shot_times_list
        self.band_times_list = band_times_list
        self.server_times_list = server_times_list
        self.serverlist = serverlist
        self.max_runs = max_runs  # 最大実行回数
        self.run_count = 0        # 現在の実行回数
        self.flag = flag
    
    def run(self):
        global global_finished_bell, global_finished_correction
        qubit_initialised = False
        entanglement_ready = False
        c_meas_results = None
        qubit_init_program = InitStateProgram()
        measure_program = BellMeasurementProgram()
        server = ServerProgram()
        m5xserver = M5XServerProgram()
        collect1 = Collect1Program()
        collect2 = Collect2Program()
        xmesure = XmeasureProgram()
       
        init = True
        b1 = []
        b2 = []
        b3 = []
        b4 = []
        b5 = []
        astate=0
        start_time = ns.sim_time()
        # print(f"スタート: {ns.sim_time()}")
        self.node.qmemory.execute_program(qubit_init_program)
        
        while True:
            # if self.run_count == self.max_runs:
            #         ns.sim_stop()
            if not init:
                yield self.await_timer(10)
                start_time = ns.sim_time()
                # print(f"スタート: {ns.sim_time()}")
                self.node.qmemory.execute_program(qubit_init_program)
                init = True
                qubit_index_4 = 3
                qubit_4, = self.node.qmemory.pop(qubit_index_4)
                qapi.discard(qubit_4)
            expr = yield (self.await_program(self.node.qmemory) |
                          self.await_port_input(self.node.ports["qin_Alice"]))  # 修正箇所
            if expr.first_term.value:
                qubit_initialised = True
                # print(f"初期化終了: {ns.sim_time()}")
                s1 = ns.sim_time() - start_time
            else:
                entanglement_ready = True
                # print(f"1個目のもつれ到達: {ns.sim_time()}")
                b1.append(ns.sim_time())
            if qubit_initialised and entanglement_ready:
                # print(f"測定開始: {ns.sim_time()}")
                qubit_initialised = False
                entanglement_ready = False
                e1 = ns.sim_time()
                yield self.node.qmemory.execute_program(measure_program)
                s2 = ns.sim_time() - e1
                # print(f"測定終了: {ns.sim_time()}")
                m3, = measure_program.output["M3"]
                m4, = measure_program.output["M4"]
                # ポップして破棄
                qubit_index_3 = 2
                qubit_index_4 = 3
                qubit_4, = self.node.qmemory.pop(qubit_index_4)
                qapi.discard(qubit_4)
                # print("bob測定1:", m3, m4)
                self.node.ports["cout_bob"].tx_output((m3, m4))
                # print(f"測定結果送信: {ns.sim_time()}")
                astate=1
            while astate==1:
                expr = yield (self.await_port_input(self.node.ports["cin_bob"])|
                                self.await_port_input(self.node.ports["qin_Alice"]))
                if expr.first_term.value:
                    c_meas_results, = self.node.ports["cin_bob"].rx_input().items
                    # print(f"クライアント測定完了合図古典通信受信: {ns.sim_time()}")
                else:
                    entanglement_ready = True
                    # print(f"2個目のもつれ到達: {ns.sim_time()}",entanglement_ready)
                    b2.append(ns.sim_time())
                if c_meas_results is not None:
                    # print(f"collect1開始 {ns.sim_time()}")
                    e2 = ns.sim_time()
                    # memory_status(self.node)
                    self.node.qmemory.execute_program(collect1)
                    astate=10
            while astate==10:
                expr = yield (self.await_program(self.node.qmemory) |
                                self.await_port_input(self.node.ports["qin_Alice"])) 
                if expr.first_term.value:
                    qubit_initialised = True
                    # print(f"collect1完了 {ns.sim_time()}",qubit_initialised)
                    s3 = ns.sim_time() - e2
                    # print(entanglement_ready)

                else:
                    entanglement_ready = True
                    # print(f"2個目のもつれ到達: {ns.sim_time()}",entanglement_ready)
                    b2.append(ns.sim_time())
                if qubit_initialised and entanglement_ready:
                    qubit_initialised = False
                    entanglement_ready = False
                    # print(f"測定開始(二周目へ): {ns.sim_time()}")
                    e3 = ns.sim_time()
                    yield self.node.qmemory.execute_program(measure_program)
                    s4 = ns.sim_time() - e3
                    # print(f"測定終了: {ns.sim_time()}")
                    m3, = measure_program.output["M3"]
                    m4, = measure_program.output["M4"]
                    qubit_index_3 = 2
                    qubit_index_4 = 3
                    qubit_4, = self.node.qmemory.pop(qubit_index_4)
                    qapi.discard(qubit_4)
                    # print("bob測定2:", m3, m4)
                    self.node.ports["cout_bob"].tx_output((m3, m4))
                    astate=2
            while astate==2:
                expr = yield (self.await_port_input(self.node.ports["cin_bob"])|
                                self.await_port_input(self.node.ports["qin_Alice"]))
                if expr.first_term.value:
                    c_meas_results, = self.node.ports["cin_bob"].rx_input().items
                else:
                    entanglement_ready = True
                    # print(f"3個目のもつれ到達: {ns.sim_time()}",entanglement_ready)
                    b3.append(ns.sim_time())
                if c_meas_results is not None:
                    # print(c_meas_results)
                    e4 = ns.sim_time()
                    self.node.qmemory.execute_program(collect1)
                    astate=11
            while astate==11:
                expr = yield (self.await_program(self.node.qmemory) |
                                self.await_port_input(self.node.ports["qin_Alice"]))  # 修正箇所
                if expr.first_term.value:
                    qubit_initialised = True
                    s5 = ns.sim_time() - e4
                else:
                    entanglement_ready = True
                    # print(f"3個目のもつれ到達: {ns.sim_time()}")
                    b3.append(ns.sim_time())
                if qubit_initialised and entanglement_ready:
                    qubit_initialised = False
                    entanglement_ready = False
                    e5 = ns.sim_time()
                    yield self.node.qmemory.execute_program(measure_program)
                    s6 = ns.sim_time() - e5
                    # print(f"測定終了: {ns.sim_time()}")
                    m3, = measure_program.output["M3"]
                    m4, = measure_program.output["M4"]
                    qubit_index_3 = 2
                    qubit_index_4 = 3
                    qubit_4, = self.node.qmemory.pop(qubit_index_4)
                    qapi.discard(qubit_4)
                    # print("bob測定3:", m3, m4)
                    self.node.ports["cout_bob"].tx_output((m3, m4))
                    astate=3
                    

            while astate==3:
                expr = yield (self.await_port_input(self.node.ports["cin_bob"])|
                                self.await_port_input(self.node.ports["qin_Alice"]))
                if expr.first_term.value:
                    c_meas_results, = self.node.ports["cin_bob"].rx_input().items
                else:
                    entanglement_ready = True
                    # print(f"4個目のもつれ到達: {ns.sim_time()}",entanglement_ready)
                    b4.append(ns.sim_time())

                if c_meas_results is not None:
                    e6 = ns.sim_time()
                    self.node.qmemory.execute_program(collect2)
                    astate=12
            while astate==12:
                expr = yield (self.await_program(self.node.qmemory) |
                                self.await_port_input(self.node.ports["qin_Alice"]))  # 修正箇所
                if expr.first_term.value:
                    qubit_initialised = True
                    s7 = ns.sim_time() - e6
                else:
                    entanglement_ready = True
                    # print(f"4個目のもつれ到達: {ns.sim_time()}")
                    b4.append(ns.sim_time())
                if qubit_initialised and entanglement_ready:
                    qubit_initialised = False
                    entanglement_ready = False
                    e7 = ns.sim_time()
                    yield self.node.qmemory.execute_program(measure_program)
                    s8 = ns.sim_time() - e7
                    # print(f"測定終了: {ns.sim_time()}")
                    m3, = measure_program.output["M3"]
                    m4, = measure_program.output["M4"]
                    qubit_index_3 = 2
                    qubit_index_4 = 3
                    qubit_4, = self.node.qmemory.pop(qubit_index_4)
                    qapi.discard(qubit_4)
                    # print("bob測定4:", m3, m4)
                    self.node.ports["cout_bob"].tx_output((m3, m4))
                    astate=4
            while astate==4:
                expr = yield (self.await_port_input(self.node.ports["cin_bob"])|
                                self.await_port_input(self.node.ports["qin_Alice"]))
                if expr.first_term.value:
                    c_meas_results, = self.node.ports["cin_bob"].rx_input().items
                else:
                    entanglement_ready = True
                    # print(f"5個目のもつれ到達: {ns.sim_time()}",entanglement_ready)
                    b5.append(ns.sim_time())

                if c_meas_results is not None:
                    e8 = ns.sim_time()
                    self.node.qmemory.execute_program(collect2)
                    e9 = ns.sim_time()
                    astate=13
            while astate==13:
                expr = yield (self.await_program(self.node.qmemory) |
                                self.await_port_input(self.node.ports["qin_Alice"]))  # 修正箇所
                if expr.first_term.value:
                    qubit_initialised = True
                    s9 = ns.sim_time() - e8
                else:
                    entanglement_ready = True
                    # print(f"5個目のもつれ到達: {ns.sim_time()}")
                    b5.append(ns.sim_time())
                if qubit_initialised and entanglement_ready:
                    qubit_initialised = False
                    entanglement_ready = False
                    e9 = ns.sim_time()
                    yield self.node.qmemory.execute_program(measure_program)
                    s10 = ns.sim_time() - e9
                    m3, = measure_program.output["M3"]
                    m4, = measure_program.output["M4"]
                    qubit_index_3 = 2
                    qubit_index_4 = 3
                    qubit_4, = self.node.qmemory.pop(qubit_index_3)  # 修正: インデックス
                    qapi.discard(qubit_4)
                    # print("bob測定5:", m3, m4)
                    self.node.ports["cout_bob"].tx_output((m3, m4))
                    # print(f"3 simulation time: {ns.sim_time()}")
                    astate=5
            while astate==5:
                expr = yield (self.await_port_input(self.node.ports["cin_bob"]))
                if expr.first_term.value:
                    c_meas_results, = self.node.ports["cin_bob"].rx_input().items
                if c_meas_results is not None:
                    e10 = ns.sim_time()
                    if self.flag == 1:
                        yield self.node.qmemory.execute_program(xmesure)
                    if c_meas_results == 1:
                        yield self.node.qmemory.execute_program(server)
                        m1, = server.output["M1"]
                        m2, = server.output["M2"]
                    else:
                        yield self.node.qmemory.execute_program(server)
                        m1, = server.output["M1"]
                        m2, = server.output["M2"]
                    s11 = ns.sim_time() - e10
                    # print("結果m1:", m1, "m2:", m2)
                    self.serverlist.append((m1,m2))
                    end_time = ns.sim_time()
                    run_time = end_time - start_time
                    # print(ns.sim_time())
                    # print(f"Shot run time: {run_time} ns")
                    band_time = (b1[0]-start_time) + (b2[0]-b1[-1]) + (b3[0]-b2[-1]) + (b4[0]-b3[-1]) + (b5[0]-b4[-1])
                    self.band_times_list.append(band_time)
                    self.shot_times_list.append(run_time)
                    self.server_times_list.append(s1+s2+s3+s4+s5+s6+s7+s8+s9+s10+s11)
                    b1 = []
                    b2 = []
                    b3 = []
                    b4 = []
                    b5 = []
                    self.send_signal(Signals.SUCCESS, 0)
                    self.run_count += 1
                    if self.run_count < self.max_runs:
                        qubit_initialised = False
                        entanglement_ready = False
                        init = False
                        c_meas_results = None
                        astate=0
                        qubit_index_1 = 0
                        qubit_index_2 = 1
                        qubit_index_3 = 2
                        qubit_index_4 = 3
                        # print(self.run_count)
                    else:
                        global_finished_bell = True
                        if global_finished_bell and global_finished_correction:
                            ns.sim_stop()
                        # yield self.await_timer(10000000)
                        # ns.sim_stop()




class CorrectionProtocol(NodeProtocol):
    """Bob側のプロトコル。Aliceの測定結果を受け取って必要なフィードフォワードを実行し、その後測定を行う。"""
    def __init__(self, node, clientlist,telelist,max_runs,parameter):
        super().__init__(node)
        self.clientlist = clientlist
        self.parameter = parameter
        self.telelist = telelist
        self.max_runs = max_runs  # 最大実行回数
        self.run_count = 0        # 現在の実行回数
    def run(self):
        global global_finished_bell, global_finished_correction
        port_alice = self.node.ports["cin_alice"]
        port_bob = self.node.ports["qin_Bob"]        # Bobノードの量子ポート名に合わせて修正
 
        entanglement_ready = False
        meas_results = None
        m5_result = None
        program = ClientProgram()
        state=0
        angle=self.parameter
        while True:
            expr = yield (self.await_port_input(port_alice) | self.await_port_input(port_bob))
            if expr.first_term.value:
                meas_results, = port_alice.rx_input().items
                # print(f"古典メッセージ1受け取り: {ns.sim_time()}")
                # print("一回目tele結果m1:", meas_results[0], "m2:", meas_results[1])
            else:
                entanglement_ready = True
                # print(f"bobiこめ？のもつれ供給: {ns.sim_time()}")

            if meas_results is not None and entanglement_ready:
                angle=self.parameter
                if meas_results[0] == 1:
                    angle=angle+np.pi
                if meas_results[1] == 1:
                    angle=-angle
                # print(f"パラメータ開始: {ns.sim_time()}")
                self.node.qmemory.execute_instruction(instr.INSTR_ROT_Z, [0], angle=angle)
                yield self.await_program(self.node.qmemory)
                # print(f"パラメータ完了: {ns.sim_time()}")
                yield self.node.qmemory.execute_program(program)
                # print(f"クライアント測定完了: {ns.sim_time()}")
                qubit_index_5 = 0
                qubit_5, = self.node.qmemory.pop(qubit_index_5)  # メモリから取り出す
                qapi.discard(qubit_5)
                m5, = program.output["M5"]
                m5_result = m5
                tele1 = meas_results
                # print(f"プロトコル調整用古典通信送信: {ns.sim_time()}")
                self.node.ports["cout_alice"].tx_output((m5))
                # print("1cli結果:", m5)
                entanglement_ready = False
                meas_results = None
                state=1

            while state==1:
                expr = yield (self.await_port_input(port_alice) |
                                self.await_port_input(port_bob))
                if expr.first_term.value:
                    meas_results, = port_alice.rx_input().items
                    # print(f"古典メッセージ2後: {ns.sim_time()}")
                    # print("2回目tele結果m1:", meas_results[0], "m2:", meas_results[1])
                else:
                    # print("bob2つ目のもつれget")
                    entanglement_ready = True

                if meas_results is not None and entanglement_ready:
                    angle=0
                    if m5_result == 1:
                        angle=np.pi
                    if meas_results[0] == 1:
                        angle=angle+np.pi
                    if meas_results[1] == 1:
                        angle=-angle
                    self.node.qmemory.execute_instruction(instr.INSTR_ROT_Z, angle=angle)
                    yield self.await_program(self.node.qmemory)
                    yield self.node.qmemory.execute_program(program)
                    qubit_index_5 = 0
                    qubit_5, = self.node.qmemory.pop(qubit_index_5)  # メモリから取り出す
                    qapi.discard(qubit_5)
                    m5, = program.output["M5"]
                    m5_2=m5
                    tele2 = meas_results
                    self.node.ports["cout_alice"].tx_output((m5))
                    # print("2cli結果:", m5)
                    entanglement_ready = False
                    meas_results = None
                    # print(m5)
                    state=2

            while state==2:
                expr = yield (self.await_port_input(port_alice) |
                                self.await_port_input(port_bob))
                if expr.first_term.value:
                    meas_results, = port_alice.rx_input().items
                    # print(f"古典メッセージ3後: {ns.sim_time()}")
                    # print("3回目tele結果m1:", meas_results[0], "m2:", meas_results[1])
                else:
                    entanglement_ready = True
                if meas_results is not None and entanglement_ready:
                    angle=0
                    if m5_result == 1:
                        angle=np.pi
                    if meas_results[0] == 1:
                        angle=angle+np.pi
                    if meas_results[1] == 1:
                        angle=-angle
                    self.node.qmemory.execute_instruction(instr.INSTR_ROT_Z, angle=angle)
                    yield self.await_program(self.node.qmemory)
                    yield self.node.qmemory.execute_program(program)
                    qubit_index_5 = 0
                    qubit_5, = self.node.qmemory.pop(qubit_index_5)  # メモリから取り出す
                    qapi.discard(qubit_5)
                    m5, = program.output["M5"]
                    m5_3 = m5
                    tele3 = meas_results
                    self.node.ports["cout_alice"].tx_output((m5))
                    # print("3cli結果:", m5)
                    entanglement_ready = False
                    meas_results = None
                    state=3

            while state==3:
                expr = yield (self.await_port_input(port_alice) |
                                self.await_port_input(port_bob))
                if expr.first_term.value:
                    meas_results, = port_alice.rx_input().items
                    # print(f"古典メッセージ4後: {ns.sim_time()}")
                    # print("4回目tele結果m1:", meas_results[0], "m2:", meas_results[1])
                else:
                    entanglement_ready = True
                if meas_results is not None and entanglement_ready:
                    angle=0
                    if m5_result == 1:
                        angle=0
                    if meas_results[0] == 1:
                        angle=angle+np.pi
                    if meas_results[1] == 1:
                        angle=-angle
                    self.node.qmemory.execute_instruction(instr.INSTR_ROT_Z, angle=angle)
                    yield self.await_program(self.node.qmemory)
                    yield self.node.qmemory.execute_program(program)
                    qubit_index_5 = 0
                    qubit_5, = self.node.qmemory.pop(qubit_index_5)  # メモリから取り出す
                    qapi.discard(qubit_5)
                    m5, = program.output["M5"]
                    m5_4 = m5
                    tele4 = meas_results
                    # print("5th; m5:", m5)
                    self.node.ports["cout_alice"].tx_output((m5))
                    # print("4cli結果:", m5)
                    entanglement_ready = False
                    meas_results = None
                    state=4

            while state==4:
                expr = yield (self.await_port_input(port_alice) |
                                self.await_port_input(port_bob))
                if expr.first_term.value:
                    meas_results, = port_alice.rx_input().items
                    # print(f"古典メッセージ5後: {ns.sim_time()}")
                    # print("5回目tele結果m1:", meas_results[0], "m2:", meas_results[1])
                else:
                    entanglement_ready = True
                if meas_results is not None:
                    # print(f"古典メッセージ5後: {ns.sim_time()}")
                    angle=0
                    if m5_result == 1:
                        angle=np.pi
                    if meas_results[0] == 1:
                        angle=angle+np.pi
                    if meas_results[1] == 1:
                        angle=-angle
                    self.node.qmemory.execute_instruction(instr.INSTR_ROT_Z, angle=angle)
                    yield self.await_program(self.node.qmemory)
                    yield self.node.qmemory.execute_program(program)
                    m5, = program.output["M5"]
                    qubit_index_5 = 0
                    qubit_5, = self.node.qmemory.pop(qubit_index_5)  # メモリから取り出す
                    qapi.discard(qubit_5)
                    m5_5 = m5
                    tele5 = meas_results
                    self.clientlist.append((m5_result,m5_2, m5_3, m5_4, m5_5))
                    self.telelist.append((tele1,tele2,tele3,tele4,tele5))
                    # print("5th; m5:", m5)
                    self.node.ports["cout_alice"].tx_output((m5))
                    state=5
                    self.send_signal(Signals.SUCCESS, 0)
                    self.run_count += 1
                    # print("5cli結果:", m5)
                    if self.run_count < self.max_runs:
                        qubit_initialised = False
                        entanglement_ready = False
                        init = False
                        c_meas_results = None
                        meas_results = None
                        # print("c")
                    else:
                        global_finished_correction = True
                        if global_finished_bell and global_finished_correction:
                            ns.sim_stop()

#
# 9) ネットワークを設定する関数を修正してゲート速度ファクターを受け入れるようにする
#
def example_network_setup(dephase_rate,node_distance, T1, T2_ratio,client_T1, sge, dge,gate_speed_factor=1.0,client_gate_speed_factor=1,entanglement_fidelity=1.0,client_fidelity=1):
    """ネットワークのセットアップを行う関数。ゲート速度ファクターを受け入れるように修正"""
    alice = Node("Alice", qmemory=create_processor(dephase_rate=dephase_rate,T1=T1,T2_ratio=T2_ratio,sge=sge,dge=dge, gate_speed_factor=gate_speed_factor))
    bob = Node("Bob", qmemory=create_client_processor(dephase_rate=client_fidelity,T1=client_T1,T2_ratio=T2_ratio,sge=sge,dge=dge, gate_speed_factor=client_gate_speed_factor))
    network = Network("Teleportation_network")
    network.add_nodes([alice, bob])

    # 古典接続
    c_conn = ClassicalConnection(length=node_distance)
    network.add_connection(alice, bob, connection=c_conn, label="classical",
                           port_name_node1="cout_bob", port_name_node2="cin_alice")
    c_conn_reverse = ClassicalConnection(length=node_distance)
    network.add_connection(bob, alice, connection=c_conn_reverse, label="classical_reverse",
                           port_name_node1="cout_alice", port_name_node2="cin_bob")

    # 量子チャネルの接続
    q_conn = ExternalEntanglingConnection(length=node_distance,fidelity=entanglement_fidelity)
    port_ac, port_bc = network.add_connection(
        alice, bob,
        connection=q_conn,
        label="quantum",
        port_name_node1="qin_Alice",  # Alice側ポート名を一意に
        port_name_node2="qin_Bob"     # Bob側ポート名を一意に
    )

    # Aliceの "qin_Alice" → qmemoryの 'qin3'
    alice.ports[port_ac].forward_input(alice.qmemory.ports['qin3'])
    # Bobの "qin_Bob" → qmemoryの 'qin0'
    bob.ports[port_bc].forward_input(bob.qmemory.ports['qin0'])

    return network

#
# 10) シミュレーションのセットアップ関数を修正してエンタングルメント速度ファクターを受け入れるようにする
#
def example_sim_setup(node_A, node_B, shot_times_list_alice, band_times_list_alice, server_times_list_alice,
                     max_runs,gate_speed_factor=1.0, entanglement_speed_factor=1.0,serverlist=None,clientlist=None,telelist=None,parameter=0,flag=0):
    """シミュレーションのセットアップを行う関数。ゲート速度ファクターとエンタングルメント速度ファクターを受け入れるように修正"""
    def collect_fidelity_data(evexpr):
        protocol = evexpr.triggered_events[-1].source
        mem_pos = protocol.get_signal_result(Signals.SUCCESS)
        q1, = protocol.node.qmemory.pop(mem_pos)
        q2, = protocol.node.qmemory.pop(mem_pos+1)
        fidelity = qapi.fidelity(q2, ns.qubits.ketstates.s1, squared=True)
        qapi.discard(q2)
        return {"fidelity": fidelity}

    # BellMeasurementProtocol (Alice) と CorrectionProtocol (Bob)
    protocol_alice = BellMeasurementProtocol(node_A, shot_times_list_alice, band_times_list_alice, server_times_list_alice,serverlist,max_runs,flag)
    protocol_bob = CorrectionProtocol(node_B,clientlist,telelist,max_runs,parameter=parameter)

    # ノード_Aが属するネットワークを取得
    network = node_A.supercomponent  # Teleportation_network

    # 量子接続は label="quantum" で追加
    q_conn = network.get_connection(node_A, node_B, label="quantum")

    # サブコンポーネントから QSource を取得
    qsource = q_conn.subcomponents.get("AliceQSource")

    if qsource is None:
        raise ValueError("AliceQSource が見つかりませんでした。サブコンポーネントの名前を確認してください。")

    # EXTERNALモードのQSourceを段階的にトリガーするプロトコル
    ext_source_protocol = ExternalSourceProtocol(
        node=node_A,
        qsource=qsource,  
        other_node=node_B,
        mem_pos_a=3,
        mem_pos_b=0,
        base_delay=1e9  / entanglement_speed_factor,  # オリジナルの base_delay をスケーリング
        extra_delay=1e6,        # オリジナルの extra_delay をスケーリング
        max_retries=int(1e3)
    )

    # DataCollector
    dc = DataCollector(collect_fidelity_data)
    dc.collect_on(pydynaa.EventExpression(source=protocol_alice, event_type=Signals.SUCCESS.value))

    # プロトコル開始
    ext_source_protocol.start()
    protocol_alice.start()
    protocol_bob.start()

    return protocol_alice, protocol_bob, ext_source_protocol, dc

#
# 11) 実験を実行する関数を修正してゲート速度ファクターとエンタングルメント速度ファクターを変数として扱うようにする
#
def convert_tuple_list_to_counts(tuple_list):
    """
    例: [(0,1), (1,0), (0,1), ...] のようなリストを受け取り、
    {'01': 回数, '10': 回数, ...} のカウント辞書にまとめる。
    """
    counts = {}
    for outcome_tuple in tuple_list:
        # タプル → 文字列 "xy" に変換（0,1 -> "01" など）
        outcome_str = "".join(str(bit) for bit in outcome_tuple)
        # カウントをインクリメント
        counts[outcome_str] = counts.get(outcome_str, 0) + 1
    
    return counts

def evaluate_prob_difference(counts, ideal_state='10'):
    """
    'ideal_state' の測定確率が 1.0 であるはず、としたときの
    「確率がどれだけ理想値(=1)とズレているか」を返す
    """
    total_shots = sum(counts.values())
    prob_ideal = counts.get(ideal_state, 0) / total_shots
    # 真の値1.0からのズレ
    return abs(prob_ideal - 1.0)

import itertools

def run_experiment(num_runs, 
                   dephase_rates,client_fidelitys, distances, T1s, client_T1s, T2_ratios, 
                   sges, dges, gate_speed_factors, client_gate_speed_factors,
                   entanglement_fidelities, entanglement_speed_factors,max_runs,angle,flag):
    global global_finished_bell, global_finished_correction
    """
    多くのパラメータを変化させてシミュレーションを実行し、結果を収集する関数。

    Parameters
    ----------
    num_runs : int
        各パラメータ組み合わせでのシミュレーション回数。
    dephase_rates : list of float
        減衰率のリスト。
    distances : list of float
        距離のリスト（km）。
    T1s : list of float
        T1時間のリスト（ns）。
    T2s : list of float
        T2時間のリスト（ns）。
    sges : list of float
        単量子ゲートの誤り率のリスト。
    dges : list of float
        2量子ゲートの誤り率のリスト。
    gate_speed_factors : list of float
        ゲート速度ファクターのリスト。
    entanglement_fidelities : list of float
        エンタングルメントの忠実度のリスト。
    entanglement_speed_factors : list of float
        エンタングルメント速度ファクターのリスト。

    Returns
    -------
    pandas.DataFrame
        収集されたすべてのデータを含むデータフレーム。
    """
    # 結果を保存するリスト
    results = []

    # パラメータのすべての組み合わせを生成
    parameter_combinations = list(itertools.product(
        dephase_rates, distances, T1s, T2_ratios,client_T1s,client_fidelitys,client_gate_speed_factors, 
        sges, dges, gate_speed_factors, 
        entanglement_fidelities, entanglement_speed_factors
    ))

    # total_combinations = len(parameter_combinations)
    # print(f"総パラメータ組み合わせ数: {total_combinations}")

    for idx, (dephase_rate, distance, T1, T2_ratio, client_T1,client_fidelity,client_gate_speed_factor,
              sge, dge, gate_factor, 
              entanglement_fidelity, entanglement_factor) in enumerate(parameter_combinations, 1):
        # print(f"\nシミュレーション {idx}/{total_combinations}:")
        # print(f"dephase_rate={dephase_rate}, distance={distance} km, T1={T1} ns, T2={T2} ns, "
        #       f"sge={sge}, dge={dge}, gate_factor={gate_factor}, "
        #       f"entanglement_fidelity={entanglement_fidelity}, entanglement_factor={entanglement_factor}")

        # 古典通信時間を距離に基づいて計算
        cc_time = (1000 * distance / 200000 * 1e6)  # 光ファイバー中の光速を200,000 km/sと仮定

        # シミュレーションをリセット
        ns.sim_reset()
        network = example_network_setup(dephase_rate, distance, T1, T2_ratio,client_T1, sge, dge, 
                                       gate_speed_factor=gate_factor, 
                                       client_gate_speed_factor=client_gate_speed_factor,
                                       entanglement_fidelity=entanglement_fidelity,
                                       client_fidelity=client_fidelity)

        node_a = network.get_node("Alice")
        node_b = network.get_node("Bob")
        shot_times_list_alice = []
        band_times_list_alice = []
        server_times_list_alice = []
        serverlist = []
        clientlist = []
        telelist =[]

        protocol_alice, protocol_bob, ext_source_protocol, dc = \
            example_sim_setup(node_a, node_b,
                              shot_times_list_alice, band_times_list_alice, 
                              server_times_list_alice,
                              gate_speed_factor=gate_factor, 
                              entanglement_speed_factor=entanglement_factor,
                              serverlist=serverlist,
                              clientlist=clientlist,
                              telelist=telelist,
                              max_runs=max_runs,
                              parameter=angle,
                              flag=flag)

        # 指定された回数分シミュレーションを実行
        ns.sim_run(1e12)
        global_finished_bell = False
        global_finished_correction = False

        # 測定結果の処理
        if flag==0:
            for i in range(len(clientlist)):
                c = clientlist[i]
                s0, s1 = serverlist[i]
                # print(s0,s1,c)
                # clientの2番目(インデックス1)が1の時、serverの1番目(s0)を反転
                if c[2] == 1:
                    s0 = 1 - s0  # 0 → 1、 1 → 0 にトグルします

                # clientの4番目(インデックス3)が1の時、serverの2番目(s1)を反転
                if c[4] == 1:
                    s1 = 1 - s1

                # 反転した結果をserverlistに更新
                
                serverlist[i] = (s0, s1)
            # print(serverlist[i])
        if flag==1:
            for i in range(len(clientlist)):
                c = clientlist[i]
                s0, s1 = serverlist[i]
                t=telelist[i]
                # clientの2番目(インデックス1)が1の時、serverの1番目(s0)を反転
                if c[1] == 1:
                    s0 = 1 - s0

                # clientの4番目(インデックス3)が1の時、serverの2番目(s1)を反転
                if c[3] == 1:
                    s1 = 1 - s1

                # clientの5番目(インデックス4)が1の時、s0 と s1 の両方を反転
                if c[0] == 1:
                    s0 = 1 - s0
                    s1 = 1 - s1
                #t[1]=1katuc[1]==1の時
                # if c[0]==1 and t[0][1]==1:
                #     s0 = 1 - s0
                #     s1 = 1 - s1
                # if c[1]==1 and t[1][1]==1:
                #     s0 = 1 - s0
                # if c[2]==1 and t[2][1]==1:
                #     s0 = 1 - s0
                # if c[3]==1 and t[3][1]==1:
                #     s1 = 1 - s1
                # if c[4]==1 and t[4][1]==1:
                #     s1 = 1 - s1

                serverlist[i] = (s0, s1)

        # Fidelityデータを収集
        df_fidelity = dc.dataframe
        df_fidelity['distance'] = distance
        df_fidelity['gate_speed_factor'] = gate_factor
        df_fidelity['entanglement_speed_factor'] = entanglement_factor

        #     # --- シミュレーション終了直前のデバッグ出力例 ---
        # print("---- Debug: 各データリストの長さ ----")
        # print("shot_times_list_alice の長さ:", len(shot_times_list_alice))
        # print("band_times_list_alice の長さ:", len(band_times_list_alice))
        # print("server_times_list_alice の長さ:", len(server_times_list_alice))
        # print("serverlist の長さ:", len(serverlist))
        # print("clientlist の長さ:", len(clientlist))
        # print("telelist の長さ:", len(telelist))

        # 時間データを収集
        shot_times_df = pandas.DataFrame({
            "distance": [distance] * len(shot_times_list_alice),
            "gate_speed_factor": [gate_factor] * len(shot_times_list_alice),
            "entanglement_speed_factor": [entanglement_factor] * len(shot_times_list_alice),
            "execution_time": shot_times_list_alice
        })
        band_times_df = pandas.DataFrame({
            "distance": [distance] * len(band_times_list_alice),
            "gate_speed_factor": [gate_factor] * len(band_times_list_alice),
            "entanglement_speed_factor": [entanglement_factor] * len(band_times_list_alice),
            "band_time": band_times_list_alice
        })
        server_times_df = pandas.DataFrame({
            "distance": [distance] * len(server_times_list_alice),
            "gate_speed_factor": [gate_factor] * len(server_times_list_alice),
            "entanglement_speed_factor": [entanglement_factor] * len(server_times_list_alice),
            "server_time": server_times_list_alice
        })
        meas_df = pandas.DataFrame({
            "distance": [distance] * len(server_times_list_alice),
            "gate_speed_factor": [gate_factor] * len(server_times_list_alice),
            "entanglement_speed_factor": [entanglement_factor] * len(server_times_list_alice),
            "server_result": serverlist,
            "client_result": clientlist
        })

        # counts_resultの作成
        counts_result = convert_tuple_list_to_counts(serverlist)
        diff_prob = evaluate_prob_difference(counts_result, ideal_state='10')

        # 結果の保存
        for i in range(len(serverlist)):
            m = serverlist[i]
            c = clientlist[i]
            t = telelist[i]
            result = {
                "dephase_rate": dephase_rate,
                "client_fidelity": client_fidelity,
                "distance": distance,
                "T1": T1,
                "T2_ratio": T2_ratio,
                "client_T1": client_T1,
                "sge": sge,
                "dge": dge,
                "gate_speed_factor": gate_factor,
                "client_gate_speed_factor": client_gate_speed_factor,
                "entanglement_fidelity": entanglement_fidelity,
                "entanglement_speed_factor": entanglement_factor,
                "shot_time": shot_times_list_alice[i],
                "band_time": band_times_list_alice[i],
                "server_time": server_times_list_alice[i],
                "cc_time": cc_time,
                "server_result": m,
                "client_result": c,
                "teleport_result": t,
                "diff_prob": diff_prob,
                "angle" : angle
            }
            results.append(result)

        # print(f"シミュレーション完了: diff_prob={diff_prob}")

    # データフレームに変換
    results_df = pandas.DataFrame(results)
    

    return results_df



def count_tuple_frequencies(data):
    freq_dict = {}
    for pair in data:
        key = "".join(map(str, pair[::-1]))  # 前後を反転
        if key in freq_dict:
            freq_dict[key] += 1
        else:
            freq_dict[key] = 1
    return freq_dict

def count_tuple_frequencies(data):
    freq_dict = {}
    for pair in data:
        key = "".join(map(str, pair[::-1]))  # 前後を反転
        if key in freq_dict:
            freq_dict[key] += 1
        else:
            freq_dict[key] = 1
    return freq_dict

def calculate_Z_cost(ans,shots):
    cost = 0
    #1
    cost += shots*(-0.4804)
    #Z0
    cost += (-ans.get("10",0)+ans.get("01",0)-ans.get("11",0)+ans.get("00",0))*(0.3435)
    #Z1
    cost += (ans.get("10",0)-ans.get("01",0)-ans.get("11",0)+ans.get("00",0))*(-0.4347)
    #Z0Z1
    cost += (-ans.get("10",0)-ans.get("01",0)+ans.get("11",0)+ans.get("00",0))*(0.5716)
    cost=cost/shots

    return cost

def calculate_X_cost(ans,shots):
    cost = 0
    cost += (-ans.get("10",0)-ans.get("01",0)+ans.get("11",0)+ans.get("00",0))*(0.0910)
    cost=cost/shots

    return cost


def ZZ_cost(num_runs, 
                   dephase_rates,client_fidelitys, distances, T1s, client_T1s, T2_ratios, 
                   sges, dges, gate_speed_factors, client_gate_speed_factors,
                   entanglement_fidelities, entanglement_speed_factors,shots,angle,flag=0):
    # print(angle)
    # print("zz hajimari")
    results_df = run_experiment(
    num_runs=num_runs,
    dephase_rates=dephase_rates,
    client_fidelitys=client_fidelitys,
    distances=distances,
    T1s=T1s,
    T2_ratios=T2_ratios,
    client_T1s=client_T1s,
    sges=sges,
    dges=dges,
    gate_speed_factors=gate_speed_factors,
    client_gate_speed_factors=client_gate_speed_factors,
    entanglement_fidelities=entanglement_fidelities,
    entanglement_speed_factors=entanglement_speed_factors,
    max_runs=shots,
    angle=angle,
    flag=flag)
    # print("zz owari")
    serverlist = results_df['server_result'].tolist()
    result = count_tuple_frequencies(serverlist)
    x=calculate_Z_cost(result,shots)
    total_shot_time = results_df['shot_time'].sum()
    return x,total_shot_time

def XX_cost(num_runs, 
                   dephase_rates,client_fidelitys, distances, T1s, client_T1s, T2_ratios, 
                   sges, dges, gate_speed_factors, client_gate_speed_factors,
                   entanglement_fidelities, entanglement_speed_factors,shots,angle,flag=1):
    results_df = run_experiment(
    num_runs=num_runs,
    dephase_rates=dephase_rates,
    client_fidelitys=client_fidelitys,
    distances=distances,
    T1s=T1s,
    T2_ratios=T2_ratios,
    client_T1s=client_T1s,
    sges=sges,
    dges=dges,
    gate_speed_factors=gate_speed_factors,
    client_gate_speed_factors=client_gate_speed_factors,
    entanglement_fidelities=entanglement_fidelities,
    entanglement_speed_factors=entanglement_speed_factors,
    max_runs=shots,
    angle=angle,
    flag=flag)
    serverlist = results_df['server_result'].tolist()
    result = count_tuple_frequencies(serverlist)
    x=calculate_X_cost(result,shots)
    total_shot_time = results_df['shot_time'].sum()
    return x,total_shot_time

# def test_cost(angle):
#     shots=100
#     num_runs = 10  # テストのため少ない回数に設定。実際の実験では増やしてください。
#     dephase_rates = [0.00]
#     # distances = list(range(100, 1001, 100))  # km
#     distances = [1000]
#     T1s = [1e160]  # ns
#     T2s = [1e150]   # ns
#     sges = [0.000]  # 単量子ゲートの誤り率
#     dges = [0.00]    # 2量子ゲートの誤り率
#     gate_speed_factors = [1.0]
#     entanglement_fidelities = [1]
#     entanglement_speed_factors = [300]
#     z,zts = ZZ_cost(
#     num_runs=num_runs,
#     dephase_rates=dephase_rates,
#     distances=distances,
#     T1s=T1s,
#     T2s=T2s,
#     sges=sges,
#     dges=dges,
#     gate_speed_factors=gate_speed_factors,
#     entanglement_fidelities=entanglement_fidelities,
#     entanglement_speed_factors=entanglement_speed_factors,
#     shots=shots,
#     angle=angle
#     )   
#     x,xts=XX_cost(num_runs=num_runs,
#     dephase_rates=dephase_rates,
#     distances=distances,
#     T1s=T1s,
#     T2s=T2s,
#     sges=sges,
#     dges=dges,
#     gate_speed_factors=gate_speed_factors,
#     entanglement_fidelities=entanglement_fidelities,
#     entanglement_speed_factors=entanglement_speed_factors,
#     shots=shots,
#     angle=angle
#     )
    # cost=z+2*x
    # angle_history.append(angle)
    # cost_history.append(cost)
    # return cost

# from scipy.optimize import minimize_scalar

# angle_history=[]
# cost_history=[]
# result = minimize_scalar(test_cost, method='bounded', bounds=(-np.pi,np.pi),tol=0.0015)

# if result.success:
#     print("エネルギー:", result.fun + 1/1.4172975)
#     print("θ:", result.x)
#     print("評価回数:", result.nfev)
#     print("反復回数:", result.nit)
# else:
#     print("最適化に失敗しました。メッセージ:", result.message)

# print(result.x)

import time
import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar


def run_vqe_optimization_experiment(
    fidelity_factors,
    num_runs,
    dephase_rates,
    client_fidelitys,
    distances,
    T1s,
    T2_ratios,
    client_T1s,
    sges,
    dges,
    gate_speed_factors,
    client_gate_speed_factors,
    entanglement_fidelities,
    entanglement_speed_factors,
    shots,
    flag,
    tol=0.0015,
    bounds=(-np.pi, np.pi)
):
    results = []
    # Apply fidelity_factors to sges and dges to create lists
    # sges = [sges[0] * factor for factor in fidelity_factors]
    # dges = [dges[0] * factor for factor in fidelity_factors]
    # パラメータの全組み合わせを生成
    
    parameter_combinations = list(itertools.product(
        dephase_rates, distances, T1s, T2_ratios, client_T1s,client_fidelitys,client_gate_speed_factors, 
        fidelity_factors, gate_speed_factors, 
        entanglement_fidelities, entanglement_speed_factors
    ))
    total_combinations = len(parameter_combinations)
    for idx, (dephase_rate, distance, T1, T2_ratio, client_T1,client_fidelity,client_gate_speed_factor,fidelity_factor, gate_factor, entanglement_fidelity, entanglement_factor) in enumerate(parameter_combinations, 1):
        sge = sges[0] * fidelity_factor
        dge = dges[0] * fidelity_factor
        print(f"\nシミュレーション {idx}/{total_combinations}:")
        print(f"dephase_rate={dephase_rate}, distance={distance} km, T1={T1} ns, T2={T1*T2_ratio} ns, "
              f"sge={sge}, dge={dge}, gate_speed_factor={gate_factor}, "
              f"entanglement_fidelity={entanglement_fidelity}, entanglement_speed_factor={entanglement_factor}")

        total_time_records = []

        def cost_func(angle):
            cost_zz, total_time_zz = ZZ_cost(
                num_runs=num_runs,
                dephase_rates=[dephase_rate],
                client_fidelitys=[client_fidelity],
                distances=[distance],
                T1s=[T1],
                T2_ratios=[T2_ratio],
                client_T1s=[client_T1],
                sges=[sge],
                dges=[dge],
                gate_speed_factors=[gate_factor],
                entanglement_fidelities=[entanglement_fidelity],
                entanglement_speed_factors=[entanglement_factor],
                client_gate_speed_factors=[client_gate_speed_factor],
                shots=shots,
                angle=angle,
                flag=0
            )
            cost_xx, total_time_xx = XX_cost(
                num_runs=num_runs,
                dephase_rates=[dephase_rate],
                client_fidelitys=[client_fidelity],
                distances=[distance],
                T1s=[T1],
                T2_ratios=[T2_ratio],
                client_T1s=[client_T1],
                sges=[sge],
                dges=[dge],
                gate_speed_factors=[gate_factor],
                entanglement_fidelities=[entanglement_fidelity],
                entanglement_speed_factors=[entanglement_factor],
                client_gate_speed_factors=[client_gate_speed_factor],
                shots=shots,
                angle=angle,
                flag=1
            )
            total_time = total_time_zz + total_time_xx
            total_time_records.append({total_time})
            return cost_zz + 2 * cost_xx

        start_time = time.perf_counter()
        result = minimize_scalar(cost_func, method='bounded', bounds=bounds, tol=tol)
        end_time = time.perf_counter()
        optimization_time = end_time - start_time

        final_energy = result.fun + 1 / 1.4172975
        final_angle = result.x
        nfev = result.nfev
        nit = result.nit
        total_time_sum = sum(next(iter(time_set)) for time_set in total_time_records)

        df = pd.DataFrame({
            'final_energy': [final_energy],
            'final_angle': [final_angle],
            'nfev': [nfev],
            'nit': [nit],
            'total_time': [total_time_sum],
            'fidelity_factors': [fidelity_factor],
            'num_runs': [num_runs],
            'dephase_rate': [dephase_rate],
            'client_fidelity': [client_fidelity],
            'distance': [distance],
            'T1': [T1],
            'T2_ratio': [T2_ratio],
            'client_T1': [client_T1],
            'sge': [sge],
            'dge': [dge],
            'gate_speed_factor': [gate_factor],
            'client_gate_speed_factor': [client_gate_speed_factor],
            'entanglement_fidelity': [entanglement_fidelity],
            'entanglement_speed_factor': [entanglement_factor],
            'shots': [shots],
            'flag': [flag]
        })
        results.append(df)

    results_df = pd.concat(results, ignore_index=True)
    return results_df

import numpy as np
import pandas as pd
import numpy as np
import time
import itertools
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import seaborn as sns
from SALib.sample import saltelli
from SALib.analyze import sobol
from sklearn.linear_model import LinearRegression
import os

# Assume importing run_vqe_optimization_experiment and cost functions
# from your_module import run_vqe_optimization_experiment, ZZ_cost, XX_cost

import numpy as np
import pandas as pd
import time
import itertools
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import seaborn as sns
from SALib.sample import saltelli
from SALib.analyze import sobol
from sklearn.linear_model import LinearRegression
import os

# Assume importing run_vqe_optimization_experiment and cost functions
# from your_module import run_vqe_optimization_experiment, ZZ_cost, XX_cost

def run_sobol_sensitivity_analysis():
    """
    Execute global sensitivity analysis based on Sobol method
    Analyze three parameters: T1/T2 coherence time, entanglement fidelity, and noise rate
    Using log-transformed sampling for all parameters
    """
    # Define the problem
    problem = {
        'num_vars': 3,
        'names': ['coherence_time', 'entanglement_fidelity', 'noise_rate'],
        # 'bounds': [[np.log10(1e11), np.log10(1e13)],      # T1 coherence time (log10 scale)
        #           [np.log10(0.0001), np.log10(0.01)],       # Entanglement fidelity (log10 scale)
        #           [np.log10(0.000001), np.log10(0.0001)]]   # Noise rate (log10 scale)
        'bounds': [[np.log10(1e10), np.log10(1e12)],      # T1 coherence time (log10 scale)
                  [np.log10(0.001), np.log10(0.1)],       # Entanglement fidelity (log10 scale)
                  [np.log10(0.00001), np.log10(0.001)]]   # Noise rate (log10 scale)
    }
    # 'bounds': [[np.log10(1e11), np.log10(1e13)],      # T1 coherence time (log10 scale)
    #               [np.log10(0.99), np.log10(0.9999)],       # Entanglement fidelity (log10 scale)
    #               [np.log10(0.000001), np.log10(0.0001)]]

    # Generate samples using Sobol method
    # N=128: Basic sample size (actual number is N*(2D+2))
    N = 64
    param_values_log = saltelli.sample(problem, N, calc_second_order=True)
    print(f"Number of parameter combinations sampled: {param_values_log.shape[0]}")
    
    # Convert log-transformed values back to original scale
    param_values = 10**param_values_log
    
    # Common parameter settings (fixed parameters)
    distances = [1000]               # km
    T2_ratios = [0.1]               # Ratio to T1
    entanglement_speed_factors = [100]
    gate_speed_factors = [1]
    client_gate_speed_factors = [1]
    fidelity_factors = [1]
    num_runs = 5                    # Set low to reduce computational cost
    shots = 500
    flag = 0                        # ZZ circuit
    tol = 0.001
    bounds = (-np.pi, np.pi)
    target_energy = -1.1615         # Target ground state energy for calculating error

    # Lists to store results
    energy_errors = []
    all_results = []

    # Run simulation for each parameter combination
    for i, params in enumerate(param_values):
        coherence_time = params[0]
        # entanglement_fidelity = params[1]
        entanglement_error = params[1]       # エラー率として取得
        entanglement_fidelity = 1 - entanglement_error   # fidelityに変換
        noise_rate = params[2]
        
        # Set parameters with the same values as specified
        T1 = coherence_time
        client_T1 = coherence_time
        dephase_rate = noise_rate
        client_fidelity = noise_rate
        sge = noise_rate
        dge = noise_rate
        
        print(f"\nシミュレーション {i+1}/{len(param_values)}:")
        print(f"coherence_time={coherence_time:.2e} ns, entanglement_fidelity={entanglement_fidelity:.6f}, "
              f"noise_rate={noise_rate:.6f}")

        # Run VQE optimization experiment
        result_df = run_vqe_optimization_experiment(
            fidelity_factors=fidelity_factors,
            num_runs=num_runs,
            dephase_rates=[dephase_rate],
            client_fidelitys=[client_fidelity],
            distances=distances,
            T1s=[T1],
            T2_ratios=T2_ratios,
            client_T1s=[client_T1],
            sges=[sge],
            dges=[dge],
            gate_speed_factors=gate_speed_factors,
            client_gate_speed_factors=client_gate_speed_factors,
            entanglement_fidelities=[entanglement_fidelity],
            entanglement_speed_factors=entanglement_speed_factors,
            shots=shots,
            flag=flag,
            tol=tol,
            bounds=bounds
        )
        
        # Extract final energy and calculate error
        final_energy = result_df['final_energy'].values[0]
        energy_error = abs(final_energy - target_energy)
        energy_errors.append(energy_error)
        
        # Save all results
        result_row = {
            'coherence_time': coherence_time,
            'entanglement_fidelity': 1-entanglement_fidelity,
            'noise_rate': noise_rate,
            'final_energy': final_energy,
            'energy_error': energy_error
        }
        all_results.append(result_row)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    # Create output folder
    output_dir = 'energy_now'
    os.makedirs(output_dir, exist_ok=True)
    
    results_df.to_csv(f'{output_dir}/energy_now.csv', index=False)
    print("シミュレーション結果をCSVに保存しました。")
    
    # For Sobol analysis, we need to pass in param_values_log (log-scale values)
    # to match the problem definition with log-scale bounds
    Si = sobol.analyze(problem, np.array(energy_errors), calc_second_order=True, print_to_console=True)
    
    # Convert sensitivity analysis results to DataFrame
    Si_df = pd.DataFrame({
        'Parameter': problem['names'],
        'S1': Si['S1'],
        'S1_conf': Si['S1_conf'],
        'ST': Si['ST'],
        'ST_conf': Si['ST_conf']
    })
    
    # Sensitivity analysis for different coherence time ranges
    coherence_ranges = [(1e10, 1e11), (1e11, 5e11), (5e11, 1e12)]
    range_results = []
    
    for c_min, c_max in coherence_ranges:
        print(f"\nコヒーレンス時間範囲 {c_min:.1e}-{c_max:.1e} nsの分析:")
        # Filter data for specific coherence time range
        range_df = results_df[(results_df['coherence_time'] >= c_min) & (results_df['coherence_time'] <= c_max)]
        
        if len(range_df) < 10:
            print(f"警告: コヒーレンス時間範囲 {c_min:.1e}-{c_max:.1e} nsのサンプル数が不足しています: {len(range_df)}")
            continue
            
        # Run multivariate regression for this range
        # Using standardized regression coefficients as a simple method
        
        # Standardize data on log scale
        X = pd.DataFrame({
            'coherence_time': np.log10(range_df['coherence_time']),
            'entanglement_fidelity': np.log10(range_df['entanglement_fidelity']),
            'noise_rate': np.log10(range_df['noise_rate'])
        })
        y = np.log10(range_df['energy_error'])
        
        X_mean = X.mean()
        X_std = X.std()
        y_mean = y.mean()
        y_std = y.std()
        
        X_norm = (X - X_mean) / X_std
        y_norm = (y - y_mean) / y_std
        
        # Linear regression using least squares
        model = LinearRegression()
        model.fit(X_norm, y_norm)
        
        # Standardized regression coefficients (sensitivity coefficients)
        coefficients = model.coef_
        
        # Contribution rates (decomposition of R² determination coefficient)
        # Note: This is a simplified method, not a complete Sobol index
        contrib = coefficients ** 2
        contrib = contrib / np.sum(contrib)
        
        range_result = {
            'coherence_range': f"{c_min:.1e}-{c_max:.1e}",
            'coherence_sensitivity': abs(coefficients[0]),
            'entanglement_sensitivity': abs(coefficients[1]),
            'noise_sensitivity': abs(coefficients[2]),
            'coherence_contribution': contrib[0],
            'entanglement_contribution': contrib[1],
            'noise_contribution': contrib[2],
            'sample_size': len(range_df)
        }
        range_results.append(range_result)
    
    # Convert results by coherence range to DataFrame
    range_df = pd.DataFrame(range_results)
    print("\nコヒーレンス時間範囲ごとの感度分析結果:")
    print(range_df)
    
    # Visualize results
    plot_sensitivity_results(Si, problem['names'], output_dir)
    plot_range_sensitivity(range_df, output_dir)
    plot_3d_surface(results_df, output_dir)
    
    return Si, Si_df, range_df, results_df

def plot_sensitivity_results(Si, param_names, output_dir):
    """Function to visualize Sobol sensitivity indices"""
    plt.figure(figsize=(10, 6))
    
    # Plot first-order (S1) and total-order (ST) sensitivity indices
    width = 0.35
    indices = np.arange(len(param_names))
    
    plt.bar(indices - width/2, Si['S1'], width, label='一次感度 (S1)')
    plt.bar(indices + width/2, Si['ST'], width, label='全次感度 (ST)')
    
    plt.xticks(indices, param_names)
    plt.xlabel('パラメータ')
    plt.ylabel('Sobol感度指標')
    plt.title('パラメータごとのSobol感度指標')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/sobol_sensitivity_indices.png', dpi=300)

def plot_range_sensitivity(range_df, output_dir):
    """Function to visualize sensitivity coefficients by coherence time range"""
    if len(range_df) == 0:
        print("警告: コヒーレンス時間範囲の結果がありません。グラフは作成されません。")
        return
        
    plt.figure(figsize=(12, 6))
    
    # Plot sensitivity coefficients
    param_names = ['コヒーレンス時間', 'エンタングルメント忠実度', 'ノイズ率']
    colors = ['#8884d8', '#82ca9d', '#ffc658']
    
    x = np.arange(len(range_df))
    width = 0.25
    
    for i, param in enumerate(['coherence_sensitivity', 'entanglement_sensitivity', 'noise_sensitivity']):
        plt.bar(x + (i-1)*width, range_df[param], width, label=f'{param_names[i]}感度', color=colors[i])
    
    plt.xlabel('コヒーレンス時間範囲 (ns)')
    plt.ylabel('感度係数 (正規化)')
    plt.title('コヒーレンス時間範囲ごとのパラメータ感度係数')
    plt.xticks(x, range_df['coherence_range'])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/coherence_range_sensitivity.png', dpi=300)
    
    # Plot contribution rates
    plt.figure(figsize=(12, 6))
    
    # Prepare data for stacked bar chart
    contribution_data = range_df[['coherence_contribution', 'entanglement_contribution', 'noise_contribution']].copy()
    
    # Create stacked bar chart
    contribution_data.plot(kind='bar', stacked=True, figsize=(12, 6), 
                          color=colors,
                          width=0.7)
    
    plt.xlabel('コヒーレンス時間範囲 (ns)')
    plt.ylabel('寄与率')
    plt.title('コヒーレンス時間範囲ごとのパラメータ寄与率')
    plt.xticks(x, range_df['coherence_range'], rotation=0)
    plt.legend(['コヒーレンス時間', 'エンタングルメント忠実度', 'ノイズ率'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/coherence_range_contribution.png', dpi=300)

def plot_3d_surface(results_df, output_dir):
    """Function to create 3D surface plots of energy error vs parameters"""
    from mpl_toolkits.mplot3d import Axes3D
    
    # First surface plot: Coherence Time vs Entanglement Fidelity
    grid_size = 20
    coherence_grid = np.logspace(np.log10(1e10), np.log10(1e12), grid_size)
    entanglement_grid = np.logspace(np.log10(0.9), np.log10(0.999), grid_size)
    C, E = np.meshgrid(coherence_grid, entanglement_grid)
    
    # Fix noise rate at median value
    noise_rate = results_df['noise_rate'].median()
    
    # Build linear regression model on log-transformed data
    X_train = pd.DataFrame({
        'coherence_time': np.log10(results_df['coherence_time']),
        'entanglement_fidelity': np.log10(results_df['entanglement_fidelity']),
        'noise_rate': np.log10(results_df['noise_rate'])
    })
    
    y_train = np.log10(results_df['energy_error'])
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict energy error on grid
    Z = np.zeros_like(C)
    for i in range(grid_size):
        for j in range(grid_size):
            c = coherence_grid[j]
            e = entanglement_grid[i]
            X_pred = pd.DataFrame({
                'coherence_time': [np.log10(c)],
                'entanglement_fidelity': [np.log10(e)],
                'noise_rate': [np.log10(noise_rate)]
            })
            Z[i, j] = 10 ** model.predict(X_pred)[0]
    
    # Create 3D surface plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(C, E, Z, cmap='viridis', linewidth=0, antialiased=True, alpha=0.8)
    
    ax.set_xlabel('コヒーレンス時間 (ns)')
    ax.set_ylabel('エンタングルメント忠実度')
    ax.set_zlabel('エネルギー誤差')
    
    ax.set_xscale('log')
    ax.set_title('コヒーレンス時間とエンタングルメント忠実度がエネルギー誤差に与える影響')
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.savefig(f'{output_dir}/3d_surface_error_vs_coherence_entanglement.png', dpi=300)
    
    # Second surface plot: Coherence Time vs Noise Rate
    noise_grid = np.logspace(np.log10(0.00001), np.log10(0.001), grid_size)
    C, N = np.meshgrid(coherence_grid, noise_grid)
    
    # Fix entanglement fidelity at median value
    entanglement_fidelity = results_df['entanglement_fidelity'].median()
    
    # Predict energy error on grid
    Z = np.zeros_like(C)
    for i in range(grid_size):
        for j in range(grid_size):
            c = coherence_grid[j]
            n = noise_grid[i]
            X_pred = pd.DataFrame({
                'coherence_time': [np.log10(c)],
                'entanglement_fidelity': [np.log10(entanglement_fidelity)],
                'noise_rate': [np.log10(n)]
            })
            Z[i, j] = 10 ** model.predict(X_pred)[0]
    
    # Create 3D surface plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(C, N, Z, cmap='viridis', linewidth=0, antialiased=True, alpha=0.8)
    
    ax.set_xlabel('コヒーレンス時間 (ns)')
    ax.set_ylabel('ノイズ率')
    ax.set_zlabel('エネルギー誤差')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('コヒーレンス時間とノイズ率がエネルギー誤差に与える影響')
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.savefig(f'{output_dir}/3d_surface_error_vs_coherence_noise.png', dpi=300)

# -*- coding: utf-8 -*-
"""
local_sobol_segment_analysis.py
--------------------------------
区間を先に定義してから、その区間ごとに Saltelli サンプリング→VQE 実験→Sobol 感度を計算し、
`visualize_3d_segments()` がそのまま読める形式（simple_segment_results.csv）で保存するスクリプト。

📌 既存の実験関数
    * run_vqe_optimization_experiment() : DataFrame を返す（final_energy 列必須）

🔧 使い方
    $ python local_sobol_segment_analysis.py  # スクリプト単体実行

生成物
    - results/local_simple_segment_results.csv
    - 同じディレクトリにフル結果やログも保存
"""

import os
import time
import numpy as np
import pandas as pd
from SALib.sample import saltelli
from SALib.analyze import sobol
from sklearn.linear_model import LinearRegression  # fallback 用

# -------------------------------------------------------------
# ★ ユーザが実装済みの関数を import してください ★
# from your_module import run_vqe_optimization_experiment
# -------------------------------------------------------------

def _midpoint_log10(low: float, high: float) -> float:
    """対数スケール範囲 [low, high] の幾何平均を返す"""
    return 10 ** ((np.log10(low) + np.log10(high)) / 2)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

import numpy as np
from typing import Sequence, Tuple, Union, List

Number = Union[int, float]

def make_log_bins(
    value_range: Tuple[Number, Number],
    n_segments: int,
    *,
    base: float = 10.0,
    include_max: bool = True,
) -> List[float]:
    """
    指定した範囲を対数スケールで n_segments 等分し、ビン境界を返す。

    Parameters
    ----------
    value_range : (min_val, max_val)
        ビン分割したい数値の（実数）範囲。 min_val > 0 である必要がある。
    n_segments : int
        生成する"区間"の数。境界点は n_segments + 1 個返る。
    base : float, default 10.0
        対数を取るときの底。自然対数でよければ np.e を指定。
    include_max : bool, default True
        True なら最大値を最後の境界点に含める。
        False の場合は最大値を超えない最小の境界点を返す。

    Returns
    -------
    List[float]
        境界点の昇順リスト（長さ n_segments+1）。
    """
    min_val, max_val = map(float, value_range)
    if min_val <= 0 or max_val <= 0:
        raise ValueError("min_val と max_val は共に 0 より大きい必要があります。")
    if min_val >= max_val:
        raise ValueError("min_val は max_val より小さくなければいけません。")
    if n_segments < 1:
        raise ValueError("n_segments は 1 以上の整数にしてください。")

    # log_{base}(value) 空間で等差に区切る
    log_min = np.log(min_val) / np.log(base)
    log_max = np.log(max_val) / np.log(base)

    # np.linspace で n_segments 等分 => n_segments+1 個
    log_edges = np.linspace(log_min, log_max, n_segments + 1)

    if not include_max:
        # 最大値を超えない最後の境界まで取得
        log_edges = log_edges[:-1]

    # base^{log_edges} -> 元のスケールへ
    return list(base ** log_edges)


# ------------------ メイン関数：区間 Sobol -------------------

def run_local_sobol_segment_analysis(
    distance_range=(1e1, 1e3),        # 距離範囲 [km]
    ent_speed_range=(1e1, 1e3),       # エンタングルメント生成速度係数範囲
    gate_speed_range=(1e0, 1e2),     # ゲート速度係数範囲
    n_dist_seg: int = 3,              # 距離区間数
    n_ent_speed_seg: int = 3,          # エンタングルメント速度区間数
    n_gate_speed_seg: int = 3,         # ゲート速度区間数
    N_local: int = 16,
    output_dir: str = "59results",
    shots: int = 100,
    num_runs: int = 5,
    flag: int = 0,
):
    """区間ごとに Sobol 感度を計算し simple_segment_results.csv を出力"""
    distance_bins = make_log_bins(distance_range, n_dist_seg)
    ent_speed_bins = make_log_bins(ent_speed_range, n_ent_speed_seg)
    gate_speed_bins = make_log_bins(gate_speed_range, n_gate_speed_seg)

    start = time.perf_counter()

    # 出力準備
    _ensure_dir(output_dir)
    simple_rows = []

    # ループ用の境界リスト
    distance_bins = list(distance_bins)
    ent_speed_bins = list(ent_speed_bins)
    gate_speed_bins = list(gate_speed_bins)

    total_segments = (len(distance_bins)-1) * (len(ent_speed_bins)-1) * (len(gate_speed_bins)-1)
    seg_counter = 0

    for i1 in range(len(distance_bins) - 1):
        for i2 in range(len(ent_speed_bins) - 1):
            for i3 in range(len(gate_speed_bins) - 1):
                seg_counter += 1
                seg_name = f"seg_{i1+1}_{i2+1}_{i3+1}"
                print(f"[{seg_counter}/{total_segments}] {seg_name} running…")

                # --- 区間 bounds を log10 で設定 ---
                bounds_log10 = [
                    [np.log10(distance_bins[i1]), np.log10(distance_bins[i1+1])],
                    [np.log10(ent_speed_bins[i2]), np.log10(ent_speed_bins[i2+1])],
                    [np.log10(gate_speed_bins[i3]), np.log10(gate_speed_bins[i3+1])],
                ]

                problem = {
                    "num_vars": 3,
                    "names": [
                        "distance",
                        "entanglement_speed_factor",
                        "gate_speed_factor",
                    ],
                    "bounds": bounds_log10,
                }

                # Saltelli サンプリング（第二次まで）
                X_log = saltelli.sample(problem, N_local, calc_second_order=False)
                X = 10 ** X_log

                total_times = []

                # ---------- VQE 実験を実行 ----------
                for row in X:
                    # Saltelli で得た 3 パラメータ
                    distance, ent_speed_factor, gate_speed_factor = row

                    # --- 本物の VQE 最適化を 1 ケースだけ回す -----------------
                    df = run_vqe_optimization_experiment(
                        fidelity_factors=[1],            # ← ゲート誤り率スケールを変えないなら 1 固定
                        num_runs=num_runs,               # ex. 5
                        dephase_rates=[0],            # メモリ dephase 率（固定値）
                        client_fidelitys=[0],         # Bob 側ゲート誤り率も同値で渡す例（固定値）
                        distances=[distance],            # [km]：Saltelli からの値
                        T1s=[1e11],                      # Alice T1（固定値）
                        T2_ratios=[0.1],                 # T2/T1 比
                        client_T1s=[1e11],               # Bob T1 = Alice と同じにする例（固定値）
                        sges=[0.0006],                     # 1qubit ゲート誤り率（固定値）
                        dges=[0.006],                     # 2qubit ゲート誤り率（固定値）
                        gate_speed_factors=[gate_speed_factor],  # ゲート速度スケール：Saltelli からの値
                        client_gate_speed_factors=[gate_speed_factor],  # Bob 側も同じ
                        entanglement_fidelities=[0.99],         # 忠実度（固定値）
                        entanglement_speed_factors=[ent_speed_factor],  # 生成速度スケール：Saltelli からの値
                        shots=shots,                     # ex. 500
                        flag=flag,                       # 0: ZZ, 1: XX
                        tol=0.015,                      # 最適化の収束閾値
                        bounds=(-np.pi, np.pi),          # θ の探索範囲
                    )

                    # total_time を取得
                    total_time = df.at[0, "total_time"]
                    total_times.append(total_time)

                # Sobol 感度計算
                Si = sobol.analyze(problem, np.array(total_times), calc_second_order=False)

                # 幾何中心点（ログミッド）
                mid_dist = _midpoint_log10(distance_bins[i1], distance_bins[i1 + 1])
                mid_ent_speed = _midpoint_log10(ent_speed_bins[i2], ent_speed_bins[i2 + 1])
                mid_gate_speed = _midpoint_log10(gate_speed_bins[i3], gate_speed_bins[i3 + 1])

                # 可視化用行を構築
                simple_rows.append(
                    {
                        "segment_id": f"{i1+1}_{i2+1}_{i3+1}",
                        "distance": mid_dist,
                        "entanglement_speed_factor": mid_ent_speed,
                        "gate_speed_factor": mid_gate_speed,
                        "target_metric": float(np.mean(total_times)),
                        "distance_contribution": float(Si["ST"][0]),
                        "entanglement_speed_factor_contribution": float(Si["ST"][1]),
                        "gate_speed_factor_contribution": float(Si["ST"][2]),
                        "use_log_scale": True,
                    }
                )

    # -------------------- CSV 出力 --------------------
    simple_df = pd.DataFrame(simple_rows)
    path_simple = os.path.join(output_dir, "new_time_now.csv")
    simple_df.to_csv(path_simple, index=False)

    elapsed = time.perf_counter() - start
    print(f"\n完了: {path_simple} に書き出しました  (elapsed {elapsed:.1f} s)")

    return path_simple, simple_df


# ---------------- スクリプト実行 -------------------
if __name__ == "__main__":
    import time

    t0 = time.perf_counter()            # ← 計測開始
    path_simple, _ = run_local_sobol_segment_analysis()   # 関数実行
    elapsed = time.perf_counter() - t0  # ← 経過時間

    print(f"\n===== 全体の実行時間: {elapsed:.1f} 秒 =====")

