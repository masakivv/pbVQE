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
import netsquid as ns, numpy as np, random, os
def initialize_seeds(base_seed: int, worker_rank: int = 0):
    """すべての RNG を同じ系列で初期化するヘルパ."""
    seed = base_seed + worker_rank     # 並列実行なら rank でずらす
    ns.set_random_state(seed)          # NetSquid 内部 RNG
    np.random.seed(seed)               # NumPy
    random.seed(seed)                  # Python 標準
    return seed

def example_sim_setup(node_A, node_B, shot_times_list_alice, band_times_list_alice, server_times_list_alice,
                     max_runs,gate_speed_factor=1.0, entanglement_speed_factor=1.0,serverlist=None,clientlist=None,telelist=None,parameter=0,flag=0):
    """シミュレーションのセットアップを行う関数。ゲート速度ファクターとエンタングルメント速度ファクターを受け入れるように修正"""
    initialize_seeds(42)
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
        ns.sim_run(1e15)
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
        dephase_rate = dephase_rates[0] * fidelity_factor
        print(dephase_rate)
        print(dge)
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
import os
from datetime import datetime

# 実験結果を保存するディレクトリ
results_dir = "all_req3"
os.makedirs(results_dir, exist_ok=True)


if __name__ == "__main__":
    # 基本設定（固定パラメータ）
    num_runs = 10
    shots = 100
    flag = 0  # 例：ZZ回路の場合
    tol = 0.01
    bounds = (-np.pi, np.pi)
    T2_ratios = [0.1]  # T1に対する比率
    
    
    # # ============= 1. dephase_ratesの実験 =============
    # dephase_rates = np.linspace(0.0, 0.01, 10).round(3).tolist()
    # client_fidelitys = [0]
    # fidelity_factors = [1]
    # distances = [500]             # km
    # T1s = [1e50]                  # ns
    # client_T1s = [1e50]
    # sges = [0]                    # 単量子ゲート誤り率
    # dges = [0]                    # 2量子ゲート誤り率
    # gate_speed_factors = [1.0]    # 理想値を1.0に設定
    # client_gate_speed_factors = [1.0]
    # entanglement_fidelities = [1]
    # entanglement_speed_factors = [100]  # 理想値を100に設定
    
    # print("実験1: デフェイズレートの実験開始")
    # df_result1 = run_vqe_optimization_experiment(
    #     num_runs=num_runs,
    #     dephase_rates=dephase_rates,
    #     client_fidelitys=client_fidelitys,
    #     fidelity_factors=fidelity_factors,
    #     distances=distances,
    #     T1s=T1s,
    #     client_T1s=client_T1s,
    #     T2_ratios=T2_ratios,
    #     sges=sges,
    #     dges=dges,
    #     gate_speed_factors=gate_speed_factors,
    #     client_gate_speed_factors=client_gate_speed_factors,
    #     entanglement_fidelities=entanglement_fidelities,
    #     entanglement_speed_factors=entanglement_speed_factors,
    #     shots=shots,
    #     flag=flag,
    #     tol=tol,
    #     bounds=bounds
    # )
    # df_result1.to_csv(f"{results_dir}/dephase_rates_experiment.csv", index=False)
    # print("実験1完了: デフェイズレート")
    
    # # ============= 2. client_fidelitysの実験 =============
    # dephase_rates = [0.00]
    # client_fidelitys = np.logspace(-1, -3, 10).round(3).tolist()
    # fidelity_factors = [1]
    # distances = [500]             # km
    # T1s = [1e50]
    # client_T1s = [1e50]
    # sges = [0]                    # 単量子ゲート誤り率
    # dges = [0]                    # 2量子ゲート誤り率
    # gate_speed_factors = [1.0]    # 理想値を1.0に設定
    # client_gate_speed_factors = [1.0]
    # entanglement_fidelities = [1]
    # entanglement_speed_factors = [100]  # 理想値を100に設定
    
    # print("実験2: クライアント忠実度の実験開始")
    # df_result2 = run_vqe_optimization_experiment(
    #     num_runs=num_runs,
    #     dephase_rates=dephase_rates,
    #     client_fidelitys=client_fidelitys,
    #     fidelity_factors=fidelity_factors,
    #     distances=distances,
    #     T1s=T1s,
    #     client_T1s=client_T1s,
    #     T2_ratios=T2_ratios,
    #     sges=sges,
    #     dges=dges,
    #     gate_speed_factors=gate_speed_factors,
    #     client_gate_speed_factors=client_gate_speed_factors,
    #     entanglement_fidelities=entanglement_fidelities,
    #     entanglement_speed_factors=entanglement_speed_factors,
    #     shots=shots,
    #     flag=flag,
    #     tol=tol,
    #     bounds=bounds
    # )
    # df_result2.to_csv(f"{results_dir}/client_fidelitys_experiment.csv", index=False)
    # print("実験2完了: クライアント忠実度")
    
    # # ============= 3. fidelity_factorsの実験 =============
    # dephase_rates = [1]
    # client_fidelitys = [0]
    # fidelity_factors = np.logspace(-1, -3, 10).round(3).tolist()
    # distances = [500]             # km
    # T1s = [1e50]
    # client_T1s = [1e50]
    # sges = [1]                    # 単量子ゲート誤り率
    # dges = [1]                    # 2量子ゲート誤り率
    # gate_speed_factors = [1.0]    # 理想値を1.0に設定
    # client_gate_speed_factors = [1.0]
    # entanglement_fidelities = [1]
    # entanglement_speed_factors = [100]  # 理想値を100に設定
    
    # print("実験3: 忠実度係数の実験開始")
    # df_result3 = run_vqe_optimization_experiment(
    #     num_runs=num_runs,
    #     dephase_rates=dephase_rates,
    #     client_fidelitys=client_fidelitys,
    #     fidelity_factors=fidelity_factors,
    #     distances=distances,
    #     T1s=T1s,
    #     client_T1s=client_T1s,
    #     T2_ratios=T2_ratios,
    #     sges=sges,
    #     dges=dges,
    #     gate_speed_factors=gate_speed_factors,
    #     client_gate_speed_factors=client_gate_speed_factors,
    #     entanglement_fidelities=entanglement_fidelities,
    #     entanglement_speed_factors=entanglement_speed_factors,
    #     shots=shots,
    #     flag=flag,
    #     tol=tol,
    #     bounds=bounds
    # )
    # df_result3.to_csv(f"{results_dir}/fidelity_factors_experiment.csv", index=False)
    # print("実験3完了: 忠実度係数")
    
    # # ============= 4. distancesの実験 =============
    # dephase_rates = [0.00]
    # client_fidelitys = [0]
    # fidelity_factors = [1]
    # distances = np.logspace(2, 4, 10).astype(int).tolist()  # 100-10000 kmを対数軸で設定
    # T1s = [1e50]
    # client_T1s = [1e50]
    # sges = [0]                    # 単量子ゲート誤り率
    # dges = [0]                    # 2量子ゲート誤り率
    # gate_speed_factors = [1.0]    # 理想値を1.0に設定
    # client_gate_speed_factors = [1.0]
    # entanglement_fidelities = [1]
    # entanglement_speed_factors = [100]  # 理想値を100に設定
    
    # print("実験4: 距離の実験開始")
    # df_result4 = run_vqe_optimization_experiment(
    #     num_runs=num_runs,
    #     dephase_rates=dephase_rates,
    #     client_fidelitys=client_fidelitys,
    #     fidelity_factors=fidelity_factors,
    #     distances=distances,
    #     T1s=T1s,
    #     client_T1s=client_T1s,
    #     T2_ratios=T2_ratios,
    #     sges=sges,
    #     dges=dges,
    #     gate_speed_factors=gate_speed_factors,
    #     client_gate_speed_factors=client_gate_speed_factors,
    #     entanglement_fidelities=entanglement_fidelities,
    #     entanglement_speed_factors=entanglement_speed_factors,
    #     shots=shots,
    #     flag=flag,
    #     tol=tol,
    #     bounds=bounds
    # )
    # df_result4.to_csv(f"{results_dir}/distances_experiment.csv", index=False)
    # print("実験4完了: 距離")
    
    # # ============= 5. T1sの実験 =============
    # dephase_rates = [0.00]
    # client_fidelitys = [0]
    # fidelity_factors = [1]
    # distances = [500]             # km
    # T1s = np.logspace(7, 11, 10).tolist()  # 1e7から1e12 nsに変更
    # client_T1s = [1e50]
    # sges = [0]                    # 単量子ゲート誤り率
    # dges = [0]                    # 2量子ゲート誤り率
    # gate_speed_factors = [1.0]    # 理想値を1.0に設定
    # client_gate_speed_factors = [1.0]
    # entanglement_fidelities = [1]
    # entanglement_speed_factors = [100]  # 理想値を100に設定
    
    # print("実験5: T1時間の実験開始")
    # df_result5 = run_vqe_optimization_experiment(
    #     num_runs=num_runs,
    #     dephase_rates=dephase_rates,
    #     client_fidelitys=client_fidelitys,
    #     fidelity_factors=fidelity_factors,
    #     distances=distances,
    #     T1s=T1s,
    #     client_T1s=client_T1s,
    #     T2_ratios=T2_ratios,
    #     sges=sges,
    #     dges=dges,
    #     gate_speed_factors=gate_speed_factors,
    #     client_gate_speed_factors=client_gate_speed_factors,
    #     entanglement_fidelities=entanglement_fidelities,
    #     entanglement_speed_factors=entanglement_speed_factors,
    #     shots=shots,
    #     flag=flag,
    #     tol=tol,
    #     bounds=bounds
    # )
    # df_result5.to_csv(f"{results_dir}/T1s_experiment.csv", index=False)
    # print("実験5完了: T1時間")
    
    # # ============= 6. client_T1sの実験 =============
    # dephase_rates = [0.00]
    # client_fidelitys = [0]
    # fidelity_factors = [1]
    # distances = [500]             # km
    # T1s = [1e50]
    # client_T1s = np.logspace(7, 12, 10).tolist()  # 1e7から1e12 nsに変更
    # sges = [0]                    # 単量子ゲート誤り率
    # dges = [0]                    # 2量子ゲート誤り率
    # gate_speed_factors = [1.0]    # 理想値を1.0に設定
    # client_gate_speed_factors = [1.0]
    # entanglement_fidelities = [1]
    # entanglement_speed_factors = [100]  # 理想値を100に設定
    
    # print("実験6: クライアントT1時間の実験開始")
    # df_result6 = run_vqe_optimization_experiment(
    #     num_runs=num_runs,
    #     dephase_rates=dephase_rates,
    #     client_fidelitys=client_fidelitys,
    #     fidelity_factors=fidelity_factors,
    #     distances=distances,
    #     T1s=T1s,
    #     client_T1s=client_T1s,
    #     T2_ratios=T2_ratios,
    #     sges=sges,
    #     dges=dges,
    #     gate_speed_factors=gate_speed_factors,
    #     client_gate_speed_factors=client_gate_speed_factors,
    #     entanglement_fidelities=entanglement_fidelities,
    #     entanglement_speed_factors=entanglement_speed_factors,
    #     shots=shots,
    #     flag=flag,
    #     tol=tol,
    #     bounds=bounds
    # )
    # df_result6.to_csv(f"{results_dir}/client_T1s_experiment.csv", index=False)
    # print("実験6完了: クライアントT1時間")
    
    # # ============= 7. sgesの実験 =============
    # dephase_rates = [0.00]
    # client_fidelitys = [0]
    # fidelity_factors = [1]
    # distances = [500]             # km
    # T1s = [1e50]
    # client_T1s = [1e50]
    # sges = np.linspace(0.0, 0.01, 10).round(3).tolist()  # 単量子ゲート誤り率
    # dges = [0]                    # 2量子ゲート誤り率
    # gate_speed_factors = [1.0]    # 理想値を1.0に設定
    # client_gate_speed_factors = [1.0]
    # entanglement_fidelities = [1]
    # entanglement_speed_factors = [100]  # 理想値を100に設定
    
    # print("実験7: 単一量子ゲート誤り率の実験開始")
    # df_result7 = run_vqe_optimization_experiment(
    #     num_runs=num_runs,
    #     dephase_rates=dephase_rates,
    #     client_fidelitys=client_fidelitys,
    #     fidelity_factors=fidelity_factors,
    #     distances=distances,
    #     T1s=T1s,
    #     client_T1s=client_T1s,
    #     T2_ratios=T2_ratios,
    #     sges=sges,
    #     dges=dges,
    #     gate_speed_factors=gate_speed_factors,
    #     client_gate_speed_factors=client_gate_speed_factors,
    #     entanglement_fidelities=entanglement_fidelities,
    #     entanglement_speed_factors=entanglement_speed_factors,
    #     shots=shots,
    #     flag=flag,
    #     tol=tol,
    #     bounds=bounds
    # )
    # df_result7.to_csv(f"{results_dir}/sges_experiment.csv", index=False)
    # print("実験7完了: 単一量子ゲート誤り率")
    
    # # ============= 8. dgesの実験 =============
    # dephase_rates = [0.00]
    # client_fidelitys = [0]
    # fidelity_factors = [1]
    # distances = [500]             # km
    # T1s = [1e50]
    # client_T1s = [1e50]
    # sges = [0]                    # 単量子ゲート誤り率
    # dges = np.linspace(0.0, 0.01, 10).round(3).tolist()  # 2量子ゲート誤り率
    # gate_speed_factors = [1.0]    # 理想値を1.0に設定
    # client_gate_speed_factors = [1.0]
    # entanglement_fidelities = [1]
    # entanglement_speed_factors = [100]  # 理想値を100に設定
    
    # print("実験8: 2量子ゲート誤り率の実験開始")
    # df_result8 = run_vqe_optimization_experiment(
    #     num_runs=num_runs,
    #     dephase_rates=dephase_rates,
    #     client_fidelitys=client_fidelitys,
    #     fidelity_factors=fidelity_factors,
    #     distances=distances,
    #     T1s=T1s,
    #     client_T1s=client_T1s,
    #     T2_ratios=T2_ratios,
    #     sges=sges,
    #     dges=dges,
    #     gate_speed_factors=gate_speed_factors,
    #     client_gate_speed_factors=client_gate_speed_factors,
    #     entanglement_fidelities=entanglement_fidelities,
    #     entanglement_speed_factors=entanglement_speed_factors,
    #     shots=shots,
    #     flag=flag,
    #     tol=tol,
    #     bounds=bounds
    # )
    # df_result8.to_csv(f"{results_dir}/dges_experiment.csv", index=False)
    # print("実験8完了: 2量子ゲート誤り率")
    
    # ============= 9. gate_speed_factorsの実験 =============
    # dephase_rates = [0.00]
    # client_fidelitys = [0]
    # fidelity_factors = [1]
    # distances = [500]             # km
    # T1s = [1e50]
    # client_T1s = [1e50]
    # sges = [0]                    # 単量子ゲート誤り率
    # dges = [0]                    # 2量子ゲート誤り率
    # gate_speed_factors = np.logspace(-1.3, 2, 10).round(3).tolist()  # 0.01倍から100倍の範囲
    # client_gate_speed_factors = [1.0]
    # entanglement_fidelities = [1]
    # entanglement_speed_factors = [100]  # 理想値を100に設定
    
    # print("実験9: ゲート速度因子の実験開始")
    # df_result9 = run_vqe_optimization_experiment(
    #     num_runs=num_runs,
    #     dephase_rates=dephase_rates,
    #     client_fidelitys=client_fidelitys,
    #     fidelity_factors=fidelity_factors,
    #     distances=distances,
    #     T1s=T1s,
    #     client_T1s=client_T1s,
    #     T2_ratios=T2_ratios,
    #     sges=sges,
    #     dges=dges,
    #     gate_speed_factors=gate_speed_factors,
    #     client_gate_speed_factors=client_gate_speed_factors,
    #     entanglement_fidelities=entanglement_fidelities,
    #     entanglement_speed_factors=entanglement_speed_factors,
    #     shots=shots,
    #     flag=flag,
    #     tol=tol,
    #     bounds=bounds
    # )
    # df_result9.to_csv(f"{results_dir}/gate_speed_factors_experiment.csv", index=False)
    print("実験9完了: ゲート速度因子")
    
    # # ============= 10. client_gate_speed_factorsの実験 =============
    # dephase_rates = [0.00]
    # client_fidelitys = [0]
    # fidelity_factors = [1]
    # distances = [500]             # km
    # T1s = [1e50]
    # client_T1s = [1e50]
    # sges = [0]                    # 単量子ゲート誤り率
    # dges = [0]                    # 2量子ゲート誤り率
    # gate_speed_factors = [1.0]    # 理想値を1.0に設定
    # client_gate_speed_factors = np.logspace(-1.3, 2, 10).round(3).tolist()  # 0.01倍から100倍の範囲
    # entanglement_fidelities = [1]
    # entanglement_speed_factors = [100]  # 理想値を100に設定
    
    # print("実験10: クライアントゲート速度因子の実験開始")
    # df_result10 = run_vqe_optimization_experiment(
    #     num_runs=num_runs,
    #     dephase_rates=dephase_rates,
    #     client_fidelitys=client_fidelitys,
    #     fidelity_factors=fidelity_factors,
    #     distances=distances,
    #     T1s=T1s,
    #     client_T1s=client_T1s,
    #     T2_ratios=T2_ratios,
    #     sges=sges,
    #     dges=dges,
    #     gate_speed_factors=gate_speed_factors,
    #     client_gate_speed_factors=client_gate_speed_factors,
    #     entanglement_fidelities=entanglement_fidelities,
    #     entanglement_speed_factors=entanglement_speed_factors,
    #     shots=shots,
    #     flag=flag,
    #     tol=tol,
    #     bounds=bounds
    # )
    # df_result10.to_csv(f"{results_dir}/client_gate_speed_factors_experiment.csv", index=False)
    # print("実験10完了: クライアントゲート速度因子")
    
    # ============= 11. entanglement_fidelitiesの実験 =============
    dephase_rates = [0.00]
    client_fidelitys = [0]
    fidelity_factors = [1]
    distances = [500]             # km
    T1s = [1e50]
    client_T1s = [1e50]
    sges = [0]                    # 単量子ゲート誤り率
    dges = [0]                    # 2量子ゲート誤り率
    gate_speed_factors = [1.0]    # 理想値を1.0に設定
    client_gate_speed_factors = [1.0]
    entanglement_fidelities = np.linspace(0.7, 1.0, 10).round(3).tolist()
    entanglement_speed_factors = [100]  # 理想値を100に設定
    
    print("実験11: エンタングルメント忠実度の実験開始")
    df_result11 = run_vqe_optimization_experiment(
        num_runs=num_runs,
        dephase_rates=dephase_rates,
        client_fidelitys=client_fidelitys,
        fidelity_factors=fidelity_factors,
        distances=distances,
        T1s=T1s,
        client_T1s=client_T1s,
        T2_ratios=T2_ratios,
        sges=sges,
        dges=dges,
        gate_speed_factors=gate_speed_factors,
        client_gate_speed_factors=client_gate_speed_factors,
        entanglement_fidelities=entanglement_fidelities,
        entanglement_speed_factors=entanglement_speed_factors,
        shots=shots,
        flag=flag,
        tol=tol,
        bounds=bounds
    )
    df_result11.to_csv(f"{results_dir}/entanglement_fidelities_experiment.csv", index=False)
    print("実験11完了: エンタングルメント忠実度")
    
    # # ============= 12. entanglement_speed_factorsの実験 =============
    # dephase_rates = [0.00]
    # client_fidelitys = [0]
    # fidelity_factors = [1]
    # distances = [500]             # km
    # T1s = [1e50]
    # client_T1s = [1e50]
    # sges = [0]                    # 単量子ゲート誤り率
    # dges = [0]                    # 2量子ゲート誤り率
    # gate_speed_factors = [1.0]    # 理想値を1.0に設定
    # client_gate_speed_factors = [1.0]
    # entanglement_fidelities = [1]
    # entanglement_speed_factors = np.logspace(1, 4, 10).astype(int).tolist()  # 10から10000を対数軸で設定
    
    # print("実験12: エンタングルメント速度因子の実験開始")
    # df_result12 = run_vqe_optimization_experiment(
    #     num_runs=num_runs,
    #     dephase_rates=dephase_rates,
    #     client_fidelitys=client_fidelitys,
    #     fidelity_factors=fidelity_factors,
    #     distances=distances,
    #     T1s=T1s,
    #     client_T1s=client_T1s,
    #     T2_ratios=T2_ratios,
    #     sges=sges,
    #     dges=dges,
    #     gate_speed_factors=gate_speed_factors,
    #     client_gate_speed_factors=client_gate_speed_factors,
    #     entanglement_fidelities=entanglement_fidelities,
    #     entanglement_speed_factors=entanglement_speed_factors,
    #     shots=shots,
    #     flag=flag,
    #     tol=tol,
    #     bounds=bounds
    # )
    # df_result12.to_csv(f"{results_dir}/entanglement_speed_factors_experiment.csv", index=False)
    # print("実験12完了: エンタングルメント速度因子")
    
    print("全ての実験が完了しました！")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# フォント設定 - 論文スタイルにはTeX風フォントが適している
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Hiragino Mincho ProN']
plt.rcParams['mathtext.fontset'] = 'stix'  # 数式のフォントも統一
plt.rcParams['font.size'] = 36  # 基本フォントサイズ
plt.rcParams['axes.labelsize'] = 40  # 軸ラベルのフォントサイズ
plt.rcParams['axes.titlesize'] = 44  # タイトルのフォントサイズ
plt.rcParams['xtick.labelsize'] = 36  # x軸目盛りのフォントサイズ
plt.rcParams['ytick.labelsize'] = 36  # y軸目盛りのフォントサイズ
plt.rcParams['legend.fontsize'] = 36  # 凡例のフォントサイズ

# 理論値の設定
THEORETICAL_ENERGY = -1.1516

# データフォルダのパス
data_folder = 'all_req3/'

# ファイルのリスト - 12の実験に対応
files = [
    # (data_folder + 'dephase_rates_experiment.csv', 'dephase_rate', 'Dephase Rate'),
    (data_folder + 'fidelity_factors_experiment.csv', 'fidelity_factors', 'Fidelity Factor'),
    (data_folder + 'gate_speed_factors_experiment.csv', 'gate_speed_factor', 'Gate Speed Factor'),
    (data_folder + 'T1s_experiment.csv', 'T1', 'T1 Relaxation Time'),
    (data_folder + 'client_fidelitys_experiment.csv', 'client_fidelity', 'Client Fidelity'),
    (data_folder + 'client_gate_speed_factors_experiment.csv', 'client_gate_speed_factor', 'Client Gate Speed Factor'),
    (data_folder + 'client_T1s_experiment.csv', 'client_T1', 'Client T1 Relaxation Time'),
    (data_folder + 'entanglement_speed_factors_experiment.csv', 'entanglement_speed_factor', 'Entanglement Bandwidth'),
    (data_folder + 'distances_experiment.csv', 'distance', 'Distance (km)'),
    # (data_folder + 'sges_experiment.csv', 'sge', 'Single-Qubit Gate Error'),
    # (data_folder + 'dges_experiment.csv', 'dge', 'Two-Qubit Gate Error'),
    
    
    (data_folder + 'entanglement_fidelities_experiment.csv', 'entanglement_fidelity', 'Entanglement Fidelity'),
    
]

# カスタムタイトル設定
custom_titles = {
    'Fidelity Factor': 'Server Fidelity',
    'Gate Speed Factor': 'Server Gate Speed Factor',
    'T1 Relaxation Time': 'Server T1 Relaxation Time',
    'Client Fidelity': 'Client Fidelity',
    'Client Gate Speed Factor': 'Client Gate Speed Factor',
    'Client T1 Relaxation Time': 'Client T1 Relaxation Time',
    'Entanglement Bandwidth': 'Entanglement Bandwidth',
    'Distance (km)': 'Distance',
    'Entanglement Fidelity': 'Entanglement Fidelity',
    # 他のパラメータも必要に応じて追加
}

# ===== 新しいX軸ラベル設定を追加 =====
custom_x_labels = {
    'Fidelity Factor': 'Server Fidelity Factor',
    'Gate Speed Factor': 'Server Gate Speed Factor',
    'T1 Relaxation Time': 'T1 Time [ns]',
    'Client Fidelity': 'Client Fidelity Scaling Factor',
    'Client Gate Speed Factor': 'Client Gate Speed Scaling Factor',
    'Client T1 Relaxation Time': 'Client T1 Time [ns]',
    'Entanglement Bandwidth': 'Entanglement Bandwidth [Hz]',
    'Distance (km)': 'Distance [km]',
    'Entanglement Fidelity': 'Entanglement Fidelity',
    # 他のパラメータも必要に応じて追加
}
# ===============================

# 対数スケールで扱うべきパラメータのリスト
log_scale_params = ['T1', 'client_T1', 'gate_speed_factor', 'client_gate_speed_factor', 'fidelity_factors', 'distance', 'entanglement_speed_factor','client_fidelity']

# Y軸の固定範囲を設定 - 新しく追加
ENERGY_Y_MIN = 0.0
ENERGY_Y_MAX = 1.5
TIME_Y_MIN = 0.0
TIME_Y_MAX = 2e4


# 各ファイルについてグラフを作成
for file_name, x_col, title in files:
    try:
        # データ読み込み
        df = pd.read_csv(file_name)
        
        # T1/client_T1グラフの特別処理
        if x_col in log_scale_params:
            # 値が科学記法の文字列の場合、数値に変換
            if isinstance(df[x_col].iloc[0], str) and 'e+' in df[x_col].iloc[0]:
                df[x_col] = df[x_col].apply(lambda x: float(x))
        
        # エネルギーの理論値からの絶対偏差を計算
        df['energy_deviation'] = np.abs(df['final_energy'] - THEORETICAL_ENERGY)
        
        # ナノ秒を秒に変換 (見やすさのため)
        df['total_time_ms'] = df['total_time'] / 1e9
        
        # 論文品質のグラフ作成
        fig, ax1 = plt.subplots(figsize=(14, 12))
        fig.subplots_adjust(right=0.80)  # 右側の余白を調整して2つ目のy軸を見やすく
        
        # 背景色を白に
        fig.patch.set_facecolor('white')
        ax1.set_facecolor('white')
        
        # 左Y軸: エネルギー偏差 (青)
        color1 = '#1f77b4'  # より落ち着いた青
        
        # タイトルと軸ラベル（英語）
        ax1.set_title(f'Effect of {title}', fontsize=44)
        # ===== 変更箇所: X軸ラベルを新しい設定から取得 =====
        x_label = custom_x_labels.get(title, f'{title}')
        ax1.set_xlabel(x_label, fontsize=40)
        # ===============================
        ax1.set_ylabel('Energy Deviation (abs)', fontsize=40, color=color1)
        
        # Y軸の範囲を固定 - 新しく追加
        ax1.set_ylim(ENERGY_Y_MIN, ENERGY_Y_MAX)
        
        # データをプロット
        energy_scatter = ax1.scatter(df[x_col], df['energy_deviation'], color=color1, marker='o', s=100, alpha=0.8)
        
        # T1/client_T1/gate_speed_factorグラフの特別処理
        column_to_use = x_col if x_col in df.columns else x_col + 's' if x_col + 's' in df.columns else x_col
        
        if x_col in log_scale_params:
            # 対数スケールでの回帰のため、元の値に対してlogを取る
            log_x = np.log10(df[column_to_use])
            z1 = np.polyfit(log_x, df['energy_deviation'], 1)
            p1 = np.poly1d(z1)
            
            # 対数空間で均等に点を生成し、元のスケールに戻す
            x_pred = np.logspace(np.log10(min(df[column_to_use])), np.log10(max(df[column_to_use])), 100)
            
            # 各x_predに対して、そのlog10値をモデルに入れて予測値を計算
            y_pred = [p1(np.log10(x)) for x in x_pred]
            
            # 修正した回帰線をプロット（コメントアウト）
            # energy_line, = ax1.plot(x_pred, y_pred, color=color1, linestyle='--', alpha=0.8, linewidth=4)
            
            # 対数スケールで表示
            ax1.set_xscale('log')
            # X軸のティックを調整
            ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0e}'))
        else:
            # 線形回帰線（エネルギー偏差）
            z1 = np.polyfit(df[column_to_use], df['energy_deviation'], 1)
            p1 = np.poly1d(z1)
            x_pred = np.linspace(min(df[column_to_use]), max(df[column_to_use]), 100)
            # energy_line, = ax1.plot(x_pred, p1(x_pred), color=color1, linestyle='--', alpha=0.8, linewidth=4)
        
        ax1.tick_params(axis='y', labelcolor=color1)
        
        # 右Y軸: 計算時間 (緑)
        ax2 = ax1.twinx()
        color2 = '#2ca02c'  # より落ち着いた緑
        ax2.set_ylabel('Computation Time (s)', fontsize=40, color=color2)
        
        ax2.set_ylim(TIME_Y_MIN, TIME_Y_MAX)
        
        # データをプロット
        time_scatter = ax2.scatter(df[column_to_use], df['total_time_ms'], color=color2, marker='s', s=100, alpha=0.8)
        
        # T1/client_T1/gate_speed_factorグラフの特別処理 - 時間についても同様に修正
        if x_col in log_scale_params:
            # 線形回帰線（時間）- 対数スケール対応
            log_x = np.log10(df[column_to_use])
            z2 = np.polyfit(log_x, df['total_time_ms'], 1)
            p2 = np.poly1d(z2)
            
            # 対数空間で均等に点を生成し、元のスケールに戻す
            x_pred = np.logspace(np.log10(min(df[column_to_use])), np.log10(max(df[column_to_use])), 100)
            
            # 各x_predに対して、そのlog10値をモデルに入れて予測値を計算
            y_pred = [p2(np.log10(x)) for x in x_pred]
            
            time_line, = ax2.plot(x_pred, y_pred, color=color2, linestyle='--', alpha=0.8, linewidth=4)
        else:
            # 線形回帰線（時間）
            z2 = np.polyfit(df[x_col], df['total_time_ms'], 1)
            p2 = np.poly1d(z2)
            x_pred = np.linspace(min(df[x_col]), max(df[x_col]), 100)
            # time_line, = ax2.plot(x_pred, p2(x_pred), color=color2, linestyle='--', alpha=0.8, linewidth=4)
        
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # グリッド線の設定（少し薄く）
        ax1.grid(True, linestyle='-', alpha=0.2, color='gray')
        
        # 枠線の設定
        for spine in ax1.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.0)
            spine.set_color('black')
        
        for spine in ax2.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.0)
            spine.set_color('black')
        
        # entanglement_fidelityの場合は表示範囲を調整
        if x_col == 'entanglement_fidelity':
            # 忠実度は通常0.9〜1.0の範囲
            min_val = max(min(df[x_col]), 0.9)
            max_val = min(max(df[x_col]), 1.0)
            ax1.set_xlim(min_val, max_val)
            ax1.ticklabel_format(axis='x', useOffset=False, style='plain')
        
        # 凡例（回帰線を除外）
        legend_elements = [
            energy_scatter, time_scatter
        ]
        
        legend_labels = [
            'Energy Deviation', 'Computation Time'
        ]
        
        ax1.legend(
            legend_elements,
            legend_labels,
            loc='best',
            frameon=True,
            fancybox=False,
            edgecolor='black',
            fontsize=36
        )
        
        # 最終調整
        plt.tight_layout()
        
        # グラフを表示
        # plt.show()
        
        print(f"Generated plot for {title}")
    except Exception as e:
        print(f"Error generating plot for {title}: {e}")

# 実際に使用するファイルだけをフィルタリング
active_files = [(f, x, t) for f, x, t in files]

# 全てのグラフを1つのページにまとめた図も作成 (3×3のグリッドに変更)
fig, axs = plt.subplots(3, 3, figsize=(30, 28))  # 3行3列で9パラメータをカバー
fig.patch.set_facecolor('white')

# 各サブプロットにデータをプロット
for i, (file_name, x_col, title) in enumerate(active_files):
    if i >= 9:  # 9つまでで終了
        break
        
    try:
        row, col = i // 3, i % 3
        ax1 = axs[row, col]
        
        # データ読み込み
        df = pd.read_csv(file_name)
        
        # エネルギーの理論値からの絶対偏差を計算
        df['energy_deviation'] = np.abs(df['final_energy'] - THEORETICAL_ENERGY)
        
        df['total_time_ms'] = df['total_time'] / 1e9
        
        # 対数スケールパラメータの特別処理
        if x_col in log_scale_params:
            # 値が科学記法の文字列の場合、数値に変換
            if isinstance(df[x_col].iloc[0], str) and 'e+' in df[x_col].iloc[0]:
                df[x_col] = df[x_col].apply(lambda x: float(x))
        
        # 左Y軸: エネルギー偏差 (青)
        color1 = '#1f77b4'
        plot_title = custom_titles.get(title, f'Effect of {title}')
        ax1.set_title(plot_title, fontsize=40)
        # ===== 変更箇所: X軸ラベルを新しい設定から取得 =====
        x_label = custom_x_labels.get(title, f'{title}')
        ax1.set_xlabel(x_label, fontsize=36)
        # ===============================
        ax1.set_ylabel('Energy Deviation [Hartree]', fontsize=36, color=color1)
        
        # Y軸の範囲を固定 - 新しく追加
        ax1.set_ylim(ENERGY_Y_MIN, ENERGY_Y_MAX)
        
        # エネルギー偏差をプロット（カラム名のエラーを防ぐため存在確認）
        column_to_use = x_col if x_col in df.columns else x_col + 's' if x_col + 's' in df.columns else x_col
        ax1.scatter(df[column_to_use], df['energy_deviation'], color=color1, marker='o', s=80, alpha=0.8)
        
        # 対数スケールパラメータの特別処理
        if x_col in log_scale_params:
            # 対数スケールでの回帰のため、元の値に対してlogを取る
            log_x = np.log10(df[column_to_use])
            z1 = np.polyfit(log_x, df['energy_deviation'], 1)
            p1 = np.poly1d(z1)
            
            # 対数空間で均等に点を生成し、元のスケールに戻す
            x_pred = np.logspace(np.log10(min(df[column_to_use])), np.log10(max(df[column_to_use])), 100)
            
            # 各x_predに対して、そのlog10値をモデルに入れて予測値を計算
            y_pred = [p1(np.log10(x)) for x in x_pred]
            
            # 対数スケールで表示
            ax1.set_xscale('log')
            # X軸のティックを調整
            ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0e}'))
            
            # 修正した回帰線をプロット（コメントアウト）
            # ax1.plot(x_pred, y_pred, color=color1, linestyle='--', alpha=0.8, linewidth=4)
        else:
            # 線形回帰線
            z1 = np.polyfit(df[column_to_use], df['energy_deviation'], 1)
            p1 = np.poly1d(z1)
            x_pred = np.linspace(min(df[column_to_use]), max(df[column_to_use]), 100)
            # ax1.plot(x_pred, p1(x_pred), color=color1, linestyle='--', alpha=0.8, linewidth=4)
        
        ax1.tick_params(axis='y', labelcolor=color1, labelsize=36)
        
        # 右Y軸: 計算時間 (緑)
        ax2 = ax1.twinx()
        color2 = '#2ca02c'
        ax2.set_ylabel('Computation Time [s]', fontsize=36, color=color2)
        
        # Y軸の範囲を固定 - 新しく追加
        
        ax2.set_ylim(TIME_Y_MIN, TIME_Y_MAX)
        
        # 時間をプロット
        ax2.scatter(df[column_to_use], df['total_time_ms'], color=color2, marker='s', s=80, alpha=0.8)
        
        # 対数スケールパラメータの特別処理 - 時間についても同様に修正
        if x_col in log_scale_params:
            # 線形回帰線（時間）- 対数スケール対応
            log_x = np.log10(df[column_to_use])
            z2 = np.polyfit(log_x, df['total_time_ms'], 1)
            p2 = np.poly1d(z2)
            
            # 対数空間で均等に点を生成し、元のスケールに戻す
            x_pred = np.logspace(np.log10(min(df[column_to_use])), np.log10(max(df[column_to_use])), 100)
            
            # 各x_predに対して、そのlog10値をモデルに入れて予測値を計算
            y_pred = [p2(np.log10(x)) for x in x_pred]
            
            # 修正した回帰線をプロット（コメントアウト）
            # ax2.plot(x_pred, y_pred, color=color2, linestyle='--', alpha=0.8, linewidth=4)
        else:
            # 線形回帰線
            z2 = np.polyfit(df[column_to_use], df['total_time_ms'], 1)
            p2 = np.poly1d(z2)
            x_pred = np.linspace(min(df[column_to_use]), max(df[column_to_use]), 100)
            # ax2.plot(x_pred, p2(x_pred), color=color2, linestyle='--', alpha=0.8, linewidth=4)
        
        ax2.tick_params(axis='y', labelcolor=color2, labelsize=36)
        
        # グリッド線
        ax1.grid(True, linestyle='-', alpha=0.2, color='gray')
        
        # 特にentanglement_fidelityの場合は表示範囲を調整
        if x_col == 'entanglement_fidelity':
            # 忠実度は通常0.9〜1.0の範囲
            min_val = max(min(df[column_to_use]), 0.9)
            max_val = min(max(df[column_to_use]), 1.0)
            ax1.set_xlim(min_val, max_val)
            ax1.ticklabel_format(axis='x', useOffset=False, style='plain')
    except Exception as e:
        print(f"Error in summary plot for {title}: {e}")

# 全体の調整
plt.suptitle('Comparison of All Quantum Hardware Parameters', fontsize=48)
fig.tight_layout(rect=[0, 0, 1, 0.97])  # タイトル用のスペースを確保
fig.savefig(data_folder + 'all_parameters_comparison.png', dpi=300, bbox_inches='tight')

# plt.show()

print("All plots displayed!")