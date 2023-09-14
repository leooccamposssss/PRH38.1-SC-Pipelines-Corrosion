import numpy as np
from scipy.stats import lognorm
import numpy as np
import random
import math
import scipy.stats as ss
import scipy
import matplotlib.pyplot as plt
from tqdm.notebook import trange, tqdm
from itertools import chain
import datetime
import pandas as pd
from dateutil.relativedelta import relativedelta
import numpy as np
from scipy.stats import lognorm


#1. Funções Auxiliares:

# poisson square wave para gerar os parametros:

def poisson_square_wave_time(): #essa função determina o intervalo de temos em que serão gerados os parâmetros ambientais.
    return math.ceil(scipy.stats.expon.rvs(loc=0, scale=1, size=1, random_state=None))

# define o estado que se encontra o gasoduto com base na corrosão:

def state_return(CDP,CLP,CRP): #Função que determina o estado em que se encontra o objeto
    state = 0
    if 0<CDP < 0.33 and CLP < 0.20 and CRP == 0:
        state = 0
    elif 0< CDP < 0.33 and 0.20< CLP < 0.40 and CRP == 0:
        state = 1
    elif 0< CDP < 0.33 and 0.40< CLP < 0.60 and CRP == 0:
        state = 2
    elif 0< CDP < 0.33 and 0.60> CLP  and CRP == 0:
        state = 3
    elif 0.33<CDP < 0.66 and CLP < 0.20 and CRP == 0:
        state = 4
    elif 0.33< CDP < 0.66 and 0.20< CLP < 0.40 and CRP == 0:
        state = 5
    elif 0.33< CDP < 0.66 and 0.40< CLP < 0.60 and CRP == 0:
        state = 6
    elif 0.33< CDP < 0.66 and 0.60> CLP  and CRP == 0:
        state = 7
    elif 0.66<CDP  and CLP < 0.20 and CRP == 0:
        state = 8
    elif 0.66< CDP  and 0.20< CLP < 0.40 and CRP == 0:
        state = 9
    elif 0.66< CDP  and 0.40< CLP < 0.60 and CRP == 0:
        state = 10
    elif 0.66< CDP  and 0.60> CLP  and CRP == 0:
        state = 11
    elif 0<CDP < 0.33 and CLP < 0.20 and CRP == 1:
        state = 12
    elif 0< CDP < 0.33 and 0.20< CLP < 0.40 and CRP == 1:
        state = 13
    elif 0< CDP < 0.33 and 0.40< CLP < 0.60 and CRP == 1:
        state = 14
    elif 0< CDP < 0.33 and 0.60> CLP  and CRP == 1:
        state = 15
    elif 0.33<CDP < 0.66 and CLP < 0.20 and CRP == 0:
        state = 16
    elif 0.33<CDP < 0.66 and 0.20< CLP < 0.40 and CRP == 0:
        state = 17
    elif 0.33<CDP < 0.66 and 0.40< CLP < 0.60 and CRP == 1:
        state = 18
    elif 0.33<CDP < 0.66 and 0.60> CLP and CRP == 1:
        state = 19
    elif 0.66<CDP  and CLP < 0.20 and CRP == 1:
        state = 20
    elif 0.66<CDP and 0.20< CLP < 0.40 and CRP == 1:
        state = 21
    elif 0.66<CDP< CLP < 0.60 and CRP == 1:
        state = 22
    elif 0.66<CDP < 0.66 and 0.60> CLP  and CRP == 1:
        state = 23
    return state

#Encontra o média e desvio padrão da lognormal para utilizar:
def get_mu(mean,std_dev):
    sigma_0 = np.sqrt(np.log(1 + (std_dev / mean)**2))
    mu_0 = np.log(mean) - 0.5 * sigma_0**2
    return np.exp(mu_0 + sigma_0**2 / 2)

def get_sigma(mean,std_dev):
    sigma_0 = np.sqrt(np.log(1 + (std_dev / mean)**2))
    mu_0 = np.log(mean) - 0.5 * sigma_0**2

    mu = np.exp(mu_0 + sigma_0**2 / 2)
    return np.sqrt(np.log(1 + (sigma_0 / mu)**2))

def CDP(corrosion_depth, maximum_corrosion_depth): #Função que calcula o CDP
    CDP = corrosion_depth/maximum_corrosion_depth
    return CDP
def CLP(corrosion_length, maximum_corrosion_length): #Função que calcula o CLP
    CLP = corrosion_length/maximum_corrosion_length
    return CLP
def linear_corr(LGR,t):
    return LGR*t/365.25

#Parâmetros:

diametro_interno = 492
diametro_externo = 509
espessura = 17
yield_stress = 2500

def papavinasam(contact_angle , water_percentage , wall_shear_stress,solids,T_c,P_t_psi, P_h2s,P_co2, so_ppm,hc03_ppm,cl_ppm):

    pre_solid = 1 if solids > 10.5 else 0

    pcr_oil = -0.33*contact_angle +55
    pcr_watter = 0.51*water_percentage +12.13
    pcr_gas =0.19*wall_shear_stress
    pcr_solid = 25*pre_solid + 50
    pcr_T = 0.57*T_c + 20
    pcr_p = -0.081*P_t_psi*14.50 +88
    pcr_h2s = 0
    pcr_PCO2 = -0.63*P_co2*14.50 +74
    pcr_so4 = 57 + -0.013*so_ppm
    pcr_hco3 = 81 + -0.014*hc03_ppm
    pcr_cl = 0.0007*cl_ppm +9.2

    pcr_adit = np.mean(pcr_oil+pcr_watter+pcr_gas+pcr_solid+pcr_T+pcr_p+pcr_h2s+pcr_PCO2+pcr_so4+pcr_hco3+pcr_cl)

    pcr_rate = (pcr_adit*11 + pcr_adit) / 12

    return (pcr_rate /360)*0.0254

def waard_milliam(pCO2, T):

    cr = math.exp(5.8 - 1710/(T) +0.671*math.log(pCO2)) +0.3

    return (cr / 360)

def parameter_simulator(): #Função para gerar os parâmetros do modelo / atualmente estão com algumas variaveis como normais para facilitar o processo

    T = float(lognorm.rvs(s=get_sigma(308,308*(0.10)), loc=0, scale=get_mu(308,308*(0.10)) )) #K
    P = float(lognorm.rvs(s=get_sigma(56.6,(56.6*0.15)), loc=0, scale=get_mu(56.6,(56.6*0.15)))) #BAR
    p_H2S = float(lognorm.rvs(s=get_sigma(3,3*(0.15)), loc=0, scale=get_mu(3,3*(0.15)))) #BAR
    p_PCO2 = float(lognorm.rvs(s=get_sigma(4,4*(0.15)), loc=0, scale=get_mu(4,4*(0.15)))) #BAR
    V = float(lognorm.rvs(s=get_sigma(5,5*(0.10)), loc=0, scale=get_mu(5,5*(0.10)))) #M/S
    pH = float(lognorm.rvs(s=get_sigma(4,4*(0.06)), loc=0, scale=get_mu(4,4*(0.06)))) #pH
    CL = float(lognorm.rvs(s=get_sigma(150,150*(0.15)), loc=0, scale=get_mu(150,150*(0.15)))) #ppm
    SO4 = float(lognorm.rvs(s=get_sigma(2000,2000*(0.15)), loc=0, scale=get_mu(2000,2000*(0.15)))) #ppm

    Rsoli = float(ss.uniform(0.5,1).rvs())
    parameter_generator = list([T,P,p_H2S,p_PCO2,V,pH,CL,SO4,Rsoli])

    return parameter_generator


#2. Função para treinamento do modelo:

def simulador_mensal(acao, tempo, defeito_d, defeito_l):

    g_vazamento = 0
    g_ruptura = 0

    para_simulacao = False

    # Salva o defeito acumulado da corrosão:
    defeito_d_acumulado = defeito_d
    defeito_l_acumulado = defeito_l


    # vamos simular para cada dia do mês:
    for dia in range(30):
        if para_simulacao == True: # a menos que
            pass
        else:
            # vamos ao que importa

            #começamos aplicando a ação da manutenção:
            custos_por_acao = [0,-13,-80,-3.5,-160]
            if dia > 13:
                fator_de_desconto = [1, 1 - 0.9487*2.718281828459045**(-0.023*((dia))), 0, 0,0]
            else:
                fator_de_desconto = [1, 1 - 0.9487*2.718281828459045**(-0.023*((dia))), 0, 1,0]

            df = fator_de_desconto[acao]


            recompensa = custos_por_acao[acao] #aplica ação de manutenção a recompensa

            if acao ==4: # Quando essa ação é aplicada, a sessão do duto é renovada:
                defeito_d_acumulado = 0
                defeito_l_acumulado = 0
                para_simulacão = True

            if dia == 0:
                parametros_simulados = parameter_simulator()
                ultimo_dia = dia
                delta_dias = poisson_square_wave_time()
            elif delta_dias + ultimo_dia < dia:
                parametros_simulados = parameter_simulator()
                ultimo_dia = dia
                delta_dias = poisson_square_wave_time()

                #Modelos de corrosão calculando a corrosão no dia

                cr = waard_milliam(parametros_simulados[3], parametros_simulados[0])

                pcr = papavinasam(135, 20, 0.05,parametros_simulados[8],parametros_simulados[0]-273,parametros_simulados[1], parametros_simulados[2], parametros_simulados[3],  parametros_simulados[7],4,parametros_simulados[6])

                defeito_d_dia = (cr + pcr)*df
                defeito_l_dia = linear_corr(0.0003, 1)*df

                #Atualizando defeito:
                defeito_l_acumulado = defeito_l_acumulado + defeito_l_dia
                defeito_d_acumulado = defeito_d_acumulado + defeito_d_dia

                #Atualizando Estado:
                if defeito_l_acumulado > defeito_d or defeito_l_acumulado > defeito_l:
                    aumentou_corrosao = 1
                else:
                    aumentou_corrosao = 0

                estado_novo = state_return(defeito_d_acumulado / espessura, defeito_l_acumulado /  linear_corr(0.0003, 40*365.25), aumentou_corrosao )

                #calculando funções limites de estados:
                #relativo ao burst:
                if math.sqrt(0.8*((defeito_l_acumulado/diametro_externo)**2)*(diametro_externo/espessura)) <= 4:
                    M = math.sqrt(1 + 0.8*((defeito_l_acumulado/diametro_externo)**2)*(diametro_externo/espessura))
                else:
                    M = 999999999999999999999999999999999999999999999999999


                A = (2*defeito_d_acumulado / espessura*3)

                pressao_explosao = (1.1*yield_stress)*(2*espessura/diametro_externo)*( (1 - A ) / (1 -A/M))

                corrosao_permitida = espessura*0.80

                #print(defeito_d_acumulado,pressao_explosao, parametros_simulados[1] )

                if pressao_explosao - parametros_simulados[1]   < 0 :

                    print(pressao_explosao,parametros_simulados[1], defeito_d_acumulado)

                    g_ruptura = 1
                    recompensa = recompensa -1760

                    para_simulacao = True

                elif corrosao_permitida - defeito_d_acumulado < 0:
                    g_vazamento = 1
                    recompensa = recompensa -480
                    para_simulacao = True

                if dia == 29 and para_simulacao == False:
                    recompensa = recompensa +7

    return recompensa, para_simulacao, defeito_d_acumulado, defeito_l_acumulado, estado_novo,g_ruptura,g_vazamento

#3. Função para treinamento:

def treinamento_q_learning(learning_rate = 0.1,reward_discount_factor = 0.95, exploration_rate = 0.95, EPISODES = 1000, MONTHS = 480):

    q_table = np.zeros((23,5))
    recompensa_por_episodio = []

    for episodio in trange(EPISODES):
        para_simulacao = False

        print(f"Episódio:  {episodio} / {EPISODES} ---")

        acoes_manutencao = []
        exploration_rate = exploration_rate - (episodio/EPISODES)*exploration_rate
        #Estado de um gasoduto novo:
        defeito_l_acumulado = 0
        defeito_d_acumulado = 0
        estado = 0
        acao = 0
        dia_da_acao = 0
        recompensa_mes = []
        t = 0

        for i in range(MONTHS):
            if i == 0:
                recompensa, para_simulacao, defeito_d_acumulado, defeito_l_acumulado, estado_novo,g_ruptura,g_vazamento = simulador_mensal(acao,t,defeito_d_acumulado,defeito_l_acumulado)
            t = t + 30
            if (acao == 2) and (dia_da_acao + 30*5*12 > t):
                acao = 2
                acao_2 = "Coating - 5 anos"
                recompensa = 7
                q_table[estado, acao] = q_table[estado, acao] + learning_rate * (recompensa + reward_discount_factor * np.max(q_table[estado_novo, :]) - q_table[estado, acao]) # realiza o q-learning
            else:
                recompensa, para_simulacao, defeito_d_acumulado, defeito_l_acumulado, estado_novo,g_ruptura,g_vazamento = simulador_mensal(acao,t,defeito_d_acumulado,defeito_l_acumulado)
                q_table[estado, acao] = q_table[estado, acao] + learning_rate * (recompensa + reward_discount_factor * np.max(q_table[estado_novo, :]) - q_table[estado, acao]) # realiza o q-learning

                acao_2 = ""
                #Exploitation /exploration:
                prob = random.uniform(0,1)
                if prob > exploration_rate:
                    acao = np.argmax(q_table[estado,:])
                    dia_da_acao = t
                else:
                    acao = random.choice([0,1,2,3,4])
                    dia_da_acao = t
            #print(recompensa)
            estado = estado_novo
            recompensa_mes.append(recompensa)
            if para_simulacao == True:
                break
        recompensa_por_episodio.append(sum(recompensa_mes))
        print (f"Recompensa: {recompensa_por_episodio[len(recompensa_por_episodio)-1]}", f"Último mês: {i}")
    return q_table, recompensa_por_episodio

#4. Função para avalair a simulação:

def testando_simulacao(tabela_q, iteracoes = 30):
    id_cenario = list(range(1,31))
    custo_manutencao = []
    vida_util = []
    custos_acao = [0,-13,-80,-3.5,-160]
    falhas_v = []
    falhas_r= []

    for episodio in trange(iteracoes):
        estado = 0
        acao = 0
        custo_M = 0
        falhas_vazamento = 0
        falhas_ruptura = 0
        defeito_d_acumulado = 0
        defeito_l_acumulado = 0
        t = 0

        for i in range(24*12):
            if (acao == 2) and (dia_da_acao + 30*5*12 > t):
                t = t +30
            else:
                acao = np.argmax(tabela_q[estado,:])
                dia_da_acao = t
                recompensa, para_simulacao, defeito_d_acumulado, defeito_l_acumulado, estado_novo,g_ruptura,g_vazamento = simulador_mensal(acao,t,defeito_d_acumulado,defeito_l_acumulado)
                custo_M = custo_M + custos_acao[acao]

                falhas_vazamento = falhas_vazamento + g_vazamento
                falhas_ruptura = falhas_ruptura + g_ruptura
                t = t +30


            if para_simulacao == True:
                break

        custo_manutencao.append(custo_M)
        falhas_v.append(falhas_vazamento)
        falhas_r.append(falhas_ruptura)
        vida_util.append(i)

    df =pd.DataFrame({"Cenário": id_cenario, "Custo de Manutenção":custo_manutencao, "Vida Útil":vida_util, "# Falhas Ruptura":falhas_r, "# Falhas Vazamento":falhas_v })
    resultados = [sum(custo_manutencao), np.mean(vida_util), sum(falhas_r), sum(falhas_v)]
    df["Custo Médio"] = df[ "Custo de Manutenção"] / df["Vida Útil"]
    return df,resultados

# Função para avaliar manutenção preventiva:

def preventina(listas):
    id_cenario = list(range(len(listas)-1))

    custos_acao = [0,-13,-80,-3.5,-160]

    custo_manutencaoS =[]
    falhas_vS = []
    falhas_rS = []
    vida_utilS = []


    for episodio in trange(len(listas)-1):

        custo_manutencao = []
        vida_util = []
        falhas_v = []
        falhas_r= []

        for k  in range(30):
            estado = 0
            acao = 0
            custo_M = 0
            falhas_vazamento = 0
            falhas_ruptura = 0
            defeito_d_acumulado = 0
            defeito_l_acumulado = 0
            t = 0

            for i in range(24*12):
                if (acao == 2) and (dia_da_acao + 30*5*12 > t):
                    t = t +30
                else:
                    acao = listas[episodio][i]
                    dia_da_acao = t
                    recompensa, para_simulacao, defeito_d_acumulado, defeito_l_acumulado, estado_novo,g_ruptura,g_vazamento = simulador_mensal(acao,t,defeito_d_acumulado,defeito_l_acumulado)
                    custo_M = custo_M + custos_acao[acao]

                    falhas_vazamento = falhas_vazamento + g_vazamento
                    falhas_ruptura = falhas_ruptura + g_ruptura
                    t = t +30
                if para_simulacao == True:
                    break

            custo_manutencao.append(custo_M)
            falhas_v.append(falhas_vazamento)
            falhas_r.append(falhas_ruptura)
            vida_util.append(i)

        custo_manutencaoS.append(sum(custo_manutencao))
        falhas_vS.append(sum(falhas_v))
        falhas_rS.append(sum(falhas_r))
        vida_utilS.append(np.mean(vida_util))


    df =pd.DataFrame({"Cenário": id_cenario, "Custo de Manutenção":custo_manutencaoS, "Vida Útil":vida_utilS, "# Falhas Ruptura":falhas_rS, "# Falhas Vazamento":falhas_vS })
    df[ "Custo de Manutenção"] = df[ "Custo de Manutenção"] / 20
    df["Custo Médio"] = df[ "Custo de Manutenção"] / df["Vida Útil"]
    return df
