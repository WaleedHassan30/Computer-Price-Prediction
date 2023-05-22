import streamlit as st
st.set_page_config(page_title='Computers Cost?', page_icon = "💻", layout = 'wide', initial_sidebar_state = 'auto')


st.success("🎨Data Science Project 💻 Computer Price Prediction")



import pandas as pd
import numpy as np
import joblib
import sklearn
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
##############################################
#### Define  Dicts
# Reading All CSV File into Variables and Create Dataframe as a Read CSV File.
from os import listdir
from os.path import isfile, join
path = "../Machine Learning/Reverse_Label\Desktop"
CSV_Files = sorted([f for f in listdir(path) if isfile(join(path, f))])
Var_Names_ = []
for i in CSV_Files:
    var = i.replace("(", "").replace(")", "").replace(".csv","").replace(" ", "_")
    locals()[var] = pd.read_csv(path+"/"+i)
    locals()[var] = locals()[var].to_numpy().tolist()
    locals()[var] = { k[0]: k[1] for k in locals()[var] }
##############################################################################
# Reading All CSV File into Variables and Create Dataframe as a Read CSV File.
from os import listdir
from os.path import isfile, join
path = "../Machine Learning/Reverse_Label/Labtop"
CSV_Files = sorted([f for f in listdir(path) if isfile(join(path, f))])
Var_Names_ = []
for i in CSV_Files:
    var = i.replace("(", "").replace(")", "").replace(".csv","").replace(" ", "_")
    locals()[var] = pd.read_csv(path+"/"+i)
    locals()[var] = locals()[var].to_numpy().tolist()
    locals()[var] = { k[0]: k[1] for k in locals()[var] }
##############################################################################
df_Desktop = pd.read_csv("CSV/Desktop.csv")
df_Labtop = pd.read_csv("CSV/Labtop.csv")

###################################

Row_Desktop_Data = ["Brand_","Color_","Date_","Product_Type_","Style_","Usage_","Weight_","CPU_Brand_","CPU_Brand_","CPU_Model_","CPU_Series_","CPU_Gen_","CPU_Cores_","CPU_Speed_","Core_Name_","GPU_Brand_","GPU_Model_","Video_Memory_","Is_SSD_","Storage_HDD_","Storage_SSD_","SSD_Type_","Memory_capacity_","Memory_Type_","Memory_Speed_","OS_Corporation_","OS_Version_","Screen_Size_","X_res_","Y_res_","Res_Type_","Touchscreen_","WideScreen_","Screen_Tec_","WebCam_","AC_Power_","Battery_Cell_","Power_Supply_W_","","Bluetooth_","Bluetooth_V_","Ethernet_","WiFi_Ver_","Thunderbolt_","Type_C_Count_","Type_A_Count_","USB_C_Ver_","PPI_"]

for i in Row_Desktop_Data:
    user_value = i
    locals()[user_value] = 0

###################################

st.sidebar.success("Predict Product Cost First.")

selection_type = st.radio("What computer are you looking for?",["Desktop","Labtop"], index=0, key=None, help=None, on_change=None, args=None, kwargs=None,disabled=False, horizontal=True, label_visibility="visible")
st.markdown("***")
if selection_type == "Desktop":
    for i in Row_Desktop_Data:
        user_value = i
        locals()[user_value] = 0
    st.header("How Much Desktop and All in One Computers Cost?")
    Brand = [key for key in Desktop_Reverse_Brand.keys()]
    Color = [key for key in Desktop_Reverse_Color.keys()]
    Date = df_Desktop.Date_First_Year.unique().astype("int")
    Style = [key for key in Desktop_Reverse_Product_Style.keys()]
    Usage = [key for key in Desktop_Reverse_Product_Usage.keys()]
    Weight_min = df_Desktop.Weight.min()
    Weight_max = df_Desktop.Weight.max()
    CPU_Brand = [key for key in Desktop_Reverse_CPU_Brand.keys()]
    CPU_Model = [key for key in Desktop_Reverse_CPU_Model.keys()] 
    CPU_Series = [key for key in Desktop_Reverse_CPU_Series.keys()] 
    CPU_Gen = [key for key in Desktop_Reverse_CPU_Gen.keys()] 
    CPU_Speed = df_Desktop.CPU_Speed.unique()
    CPU_Cores = df_Desktop.Cores.astype("str").apply(lambda x: x if x != "nan" else 6.0).unique()
    Core_Name = [key for key in Desktop_Reverse_Core_Name.keys()]
    GPU_Brand = [key for key in Desktop_Reverse_GPU_Brand.keys()]
    GPU_Model = [key for key in Desktop_Reverse_GPU_Model.keys()]
    Video_Memory = df_Desktop.Video_Memory.unique().astype("int")
    Is_SSD = ["Yes","No"]
    Storage_HDD = df_Desktop.Storage_HDD.unique().astype("int")
    Storage_SSD = df_Desktop.Storage_SSD.apply(lambda x: x * 1024 if x < 20 else x).unique()
    SSD_Type = [key for key in Desktop_Reverse_SSD_Type.keys()]
    Memory_capacity_min = df_Desktop.Memory_capacity.min().astype("int")
    Memory_capacity_max = df_Desktop.Memory_capacity.max().astype("int")
    Memory_Type = [key for key in Desktop_Reverse_Memory_Type.keys()]
    Memory_Speed = df_Desktop.Memory_Speed.unique()

    OS_Corporation = [key for key in Desktop_Reverse_OS_Corporation.keys()]
    OS_Version = [key for key in Desktop_Reverse_OS_Version.keys()]
    Screen_Size = df_Desktop.Screen_Size.apply(lambda x: x if x >= 23 else 0).unique().astype("str")
    X_res = df_Desktop.X_res.unique().astype("int")
    Y_res = df_Desktop.Y_res.unique().astype("int")
    Res_Type = [key for key in Desktop_Reverse_Res_Type.keys()]
    Touchscreen = df_Desktop.Touchscreen.apply(lambda x: "Yes" if x == 1.0 else "No").unique().tolist()
    WideScreen = df_Desktop.Touchscreen.apply(lambda x: "Yes" if x == 1.0 else "No").unique().tolist()
    Screen_Tec = [key for key in Desktop_Reverse_Screen_Tec.keys()]
    WebCam = [key for key in Desktop_Reverse_WebCam.keys()]
    AC_Power = df_Desktop.AC_Power.unique().astype("int")
    Battery_Cell = df_Desktop.Battery_Cell.unique().astype("int")
    Power_Supply_W_min = df_Desktop.AC_Power.min().astype("int")
    Power_Supply_W_max = df_Desktop.AC_Power.max().astype("int")
    Bluetooth = df_Desktop.Bluetooth.apply(lambda x: "Yes" if x == 1.0 else "No").unique()
    def Bluetooth_V(float):
        if str(float).startswith("5"):
            return 5
        elif str(float).startswith("3"):
            return 3
        elif str(float).startswith("2"):
            return 2
        else:
            return 4
    Bluetooth_V = df_Desktop.Bluetooth_V.apply(Bluetooth_V).unique()
    Ethernet = df_Desktop.Ethernet.apply(lambda x: "Yes" if x == 1.0 else "No").unique()
    WiFi_Ver = [key for key in Desktop_Reverse_WiFi_Ver.keys()]
    Thunderbolt = df_Desktop.Has_Thunderbolt.apply(lambda x: "Yes" if x == 1.0 else "No").unique()
    Type_C_Count_min = df_Desktop.Type_C_Count.unique().min()
    Type_C_Count_max = df_Desktop.Type_C_Count.unique().max()
    Type_A_Count_min = df_Desktop.Type_A_Count.unique().min()
    Type_A_Count_max = df_Desktop.Type_A_Count.unique().max()
    USB_C_Ver = [key for key in Desktop_Reverse_USB_C_Ver.keys()]
    PPI = df_Desktop.PPI.unique()

    #############################################################################################
    desktop_type = st.selectbox("Desktop Type",["Desktop" , "All in one"])
    if desktop_type == "Desktop":
        for i in Row_Desktop_Data:
            user_value = i
            locals()[user_value] = 0
        Product_Type_ = 0
        col1,col2 = st.columns(2)
        st.markdown("***")
        st.subheader("Choose product Main specifications.")
        Brand_ = col1.selectbox("What Brand do you prefer? ", Brand, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        Color_ = col2.selectbox("What Color do you prefer? ", Color, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        col1,col2 = st.columns(2)
        Date_ = col1.selectbox("Choose the date of manufacture", Date, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        Style_ = col2.selectbox("Computer Style ",Style , index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        Usage_ = st.selectbox("Product_Usage",Usage, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        Weight_ = st.number_input("Weight", min_value=Weight_min, max_value=Weight_max, step=0.25)
        st.markdown("***")
        st.subheader("Choose product CPU specifications.")
        col1,col2,col3,col4 = st.columns(4)
        CPU_Brand_ = col1.selectbox("CPU Brand", CPU_Brand, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

        if CPU_Brand_ == "Intel" or CPU_Brand_ == "Xeon":
                CPU_Model_ = col2.selectbox("CPU Model",[item for i,item in enumerate(["Core" , "Celeron" , "Xeon","Dual-Core"]) if  ["Core" , "Celeron" , "Xeon","Dual-Core"][i] in item], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

                CPU_Series_ = col3.selectbox("CPU Series ",[item for item in CPU_Series if "i" in item], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

        elif CPU_Brand_ == "AMD":
                CPU_Model_ = col2.selectbox("CPU Model",[item for i,item in enumerate(["Ryzen","Athlon"]) if  ["Ryzen","Athlon"][i] in item], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

                CPU_Series_ = col3.selectbox("CPU Series ",[item for item in CPU_Series if "Ryzen" in item], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        elif CPU_Brand_ == "Apple":
                CPU_Model_ = col2.selectbox("CPU Model",[item for i,item in enumerate(["M1"]) if  ["M1"][i] in item], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
                CPU_Series_ = "i5"


        CPU_Gen_ = col4.selectbox("CPU Gen ",CPU_Gen, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        col1,col2 = st.columns(2)
        CPU_Speed_ = col1.selectbox("CPU Speed ",CPU_Speed, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        CPU_Cores_ = col2.selectbox("CPU Cores",CPU_Cores, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        
        #####
        Core_Name_ = st.selectbox("CPU Core Name",Core_Name, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        st.markdown("***")
        st.subheader("Choose product GPU specifications.")
        col1,col2 = st.columns(2)
        GPU_Brand_ = col1.selectbox("GPU Brand", GPU_Brand, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

        if GPU_Brand_ == "Intel":
                GPU_Model_ = col2.selectbox("CPU Model",[item for i,item in enumerate(["Graphics" , "HD","UHD"]) if  ["Graphics" , "HD","UHD"][i] in item], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

        elif GPU_Brand_ == "NVIDIA":
                GPU_Model_ = col2.selectbox("CPU Model",[item for i,item in enumerate( ["GeForce","Radeon","Quadro","Iris"]) if   ["GeForce","Radeon","Quadro","Iris"][i] in item], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

        elif GPU_Brand_ == "Apple":
                GPU_Model_ = col2.selectbox("CPU Model",[item for i,item in enumerate(["M1"]) if  ["M1"][i] in item], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

        Video_Memory_ =round(float(st.number_input("Video_Memory", min_value=1, max_value=int(Video_Memory.max()))))

    
        st.markdown("***")
        st.subheader("Choose product Storage specifications.")
        col1,col2,col3 = st.columns(3)
        Is_SSD_  = col1.radio("Do You Need SSD Storage?",Is_SSD, index=0, key=None, help=None, on_change=None, args=None, kwargs=None,disabled=False, horizontal=False, label_visibility="visible")
        if Is_SSD_ == "Yes":
            Storage_SSD_ = col2.selectbox("Storage SSD ",Storage_SSD, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
            SSD_Type_ = col3.selectbox("Storage SSD Type ",SSD_Type, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
            temp = False
        else:
            Storage_HDD_ = st.selectbox("Storage HDD ",Storage_HDD, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
            temp = True

        Add_HDD = st.checkbox("Add HDD Storage", value=False, key="one", help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        if Add_HDD and Is_SSD_ == "Yes" and temp == False:
                Storage_HDD_ = st.selectbox("Storage HDD ",Storage_HDD, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")


        st.markdown("***")
        st.subheader("Choose product Memory specifications.")
        col1,col2,col3 = st.columns(3)
        Memory_capacity_ = col1.number_input("Memory_capacity", min_value=1, max_value=Memory_capacity_max, step=1)
        Memory_Type_ = col2.selectbox("Memory Type ",Memory_Type, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        Memory_Speed_ = col3.selectbox("Memory Speed",[2400.0,2666.0,2933.0,3000.0,3200.0,3600.0,4000.0,4400.0], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        st.markdown("***")
        st.subheader("Choose product Operating System specifications.")
        col1,col2 = st.columns(2)

        OS_Corporation_ = col1.selectbox("OS Corporation",OS_Corporation, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        if OS_Corporation_ == "Google":
            OS_Version_ = col2.selectbox("OS Version",[item for item in OS_Version if  'Chrome' in item.split()], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        elif OS_Corporation_ == "No Operating System":
            OS_Version_ = col2.selectbox("OS Version",[item for item in OS_Version if 'No Operating System' in item.split()], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        elif OS_Corporation_ == "Microsoft":
            OS_Version_ = col2.selectbox("OS Version",[item for item in OS_Version if 'Windows' in item.split()], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        elif OS_Corporation_ == "Apple":
            OS_Version_ = col2.selectbox("OS Version",["MacOS"], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        Screen_Size_= st.selectbox("Screen_Size",Screen_Size, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        col1,col2,col3,col4 = st.columns(4)
        st.markdown("***")
        st.subheader("Choose product Another specifications.")
        col1,col2 = st.columns(2)
        AC_Power_ = col1.number_input("AC Power" ,min_value=AC_Power.min(), max_value=AC_Power.max(), step=20)
        #if selection_Screen != "No Screen":
            #col2.selectbox("Battery Cell",Battery_Cell, index=0, key=None, help=None, on_change=None, #args=None, kwargs=None, disabled=False, label_visibility="visible")
        Power_Supply_W_ = col2.number_input("Power_Supply_W" ,min_value=Power_Supply_W_min, max_value=Power_Supply_W_max, step=100)
        col1,col2 = st.columns(2)
        Bluetooth_ = col1.radio("Do You Need Bluetooth?",Bluetooth, index=0, key=None, help=None, on_change=None, args=None, kwargs=None,disabled=False, horizontal=True, label_visibility="visible")
        Bluetooth_V_ = col2.selectbox("Bluetooth_V",Bluetooth_V, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        col1,col2,col3 = st.columns(3)
        Ethernet_ = col1.radio("Do You Need Ethernet?",Ethernet, index=0, key=None, help=None, on_change=None, args=None, kwargs=None,disabled=False, horizontal=True, label_visibility="visible")
        WiFi_Ver_ = col2.selectbox("WiFi Ver",WiFi_Ver, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        Thunderbolt_ = col3.radio("Do You Need Thunderbolt?",Thunderbolt, index=0, key=None, help=None, on_change=None, args=None, kwargs=None,disabled=False, horizontal=True, label_visibility="visible")
        col1,col2,col3 = st.columns(3)
        Type_C_Count_ = col1.number_input("Type_C_Count", min_value=Type_C_Count_min, max_value=Type_C_Count_max,step=1.0)
        Type_A_Count_ = col2.number_input("Type_A_Count", min_value=Type_A_Count_min, max_value=Type_A_Count_max,step=1.0)
        USB_C_Ver_ = col3.selectbox("USB_C_Ver",USB_C_Ver, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        PPI_ = 0
        Product_Type_ = 0
        
        Battery_Cell = 0
        st.markdown("***")
        def Replace_Yes_No(string):
            if string == "Yes":
                return 1
            else:
                return 0
        Data = {}
        Touchscreen_ = 0
        WideScreen_ = 0
        a = Is_SSD_
        b = Touchscreen_
        c = WideScreen_
        d = Bluetooth_ 
        e = Ethernet_ 
        f = Thunderbolt_ 
        g = CPU_Cores_
        Is_SSD_ =""
        Touchscreen_ =""
        WideScreen_ = ""
        Bluetooth_ = ""
        Ethernet_  = ""
        Thunderbolt_ =""
        CPU_Cores_ = g
        Is_SSD_ = Replace_Yes_No(Is_SSD_)
        Touchscreen_ = Replace_Yes_No(Touchscreen_) 
        WideScreen_ = Replace_Yes_No(WideScreen_)
        Bluetooth_ =Replace_Yes_No(Bluetooth_)
        Ethernet_ =Replace_Yes_No(Ethernet_)
        Thunderbolt_ =Replace_Yes_No(Thunderbolt_)

    else:
        for i in Row_Desktop_Data:
            user_value = i
            locals()[user_value] = 0
        Product_Type_ = 0
        st.subheader("Choose product Main specifications.")
        col1,col2 = st.columns(2)
        Brand_ = col1.selectbox("What Brand do you prefer? ", Brand, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        Color_ = col2.selectbox("What Color do you prefer? ", Color, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        col1,col2 = st.columns(2)
        Date_ = col1.selectbox("Choose the date of manufacture", Date, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        Style_ = col2.selectbox("Computer Style ",Style , index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        st.markdown("***")
        Usage_ = st.selectbox("Product_Usage",Usage, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        Weight_ = st.number_input("Weight", min_value=Weight_min, max_value=Weight_max, step=0.25)
        st.markdown("***")
        st.subheader("Choose product CPU specifications.")
        col1,col2,col3,col4 = st.columns(4)
        CPU_Brand_ = col1.selectbox("CPU Brand", CPU_Brand, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

        if CPU_Brand_ == "Intel" or CPU_Brand_ == "Xeon":
                CPU_Model_ = col2.selectbox("CPU Model",[item for i,item in enumerate(["Core" , "Celeron" , "Xeon","Dual-Core"]) if  ["Core" , "Celeron" , "Xeon","Dual-Core"][i] in item], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

                CPU_Series_ = col3.selectbox("CPU Series ",[item for item in CPU_Series if "i" in item], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

        elif CPU_Brand_ == "AMD":
                CPU_Model_ = col2.selectbox("CPU Model",[item for i,item in enumerate(["Ryzen","Athlon"]) if  ["Ryzen","Athlon"][i] in item], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

                CPU_Series_ = col3.selectbox("CPU Series ",[item for item in CPU_Series if "Ryzen" in item], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        elif CPU_Brand_ == "Apple":
                CPU_Model_ = col2.selectbox("CPU Model",[item for i,item in enumerate(["M1"]) if  ["M1"][i] in item], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
                CPU_Series_ = "i5"


        CPU_Gen_ = col4.selectbox("CPU Gen ",CPU_Gen, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        col1,col2 = st.columns(2)
        CPU_Speed_ = col1.selectbox("CPU Speed ",CPU_Speed, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        CPU_Cores_ = col2.selectbox("CPU Cores",CPU_Cores, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        ######
        Core_Name_ = st.selectbox("CPU Core Name",Core_Name, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        st.markdown("***")
        st.subheader("Choose product GPU specifications.")
        col1,col2 = st.columns(2)

        GPU_Brand_ = col1.selectbox("GPU Brand", GPU_Brand, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

        if GPU_Brand_ == "Intel":
                GPU_Model_ = col2.selectbox("CPU Model",[item for i,item in enumerate(["Graphics" , "HD","UHD"]) if  ["Graphics" , "HD","UHD"][i] in item], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

        elif GPU_Brand_ == "NVIDIA":
                GPU_Model_ = col2.selectbox("CPU Model",[item for i,item in enumerate( ["GeForce","Radeon","Quadro","Iris"]) if   ["GeForce","Radeon","Quadro","Iris"][i] in item], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

        elif GPU_Brand_ == "Apple":
                GPU_Model_ = col2.selectbox("CPU Model",[item for i,item in enumerate(["M1"]) if  ["M1"][i] in item], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        Video_Memory_ =round(float(st.number_input("Video_Memory", min_value=1, max_value=int(Video_Memory.max()))))



    
        st.markdown("***")
        st.subheader("Choose product Storage specifications.")
        col1,col2,col3 = st.columns(3)
        Is_SSD_  = col1.radio("Do You Need SSD Storage?",Is_SSD, index=0, key=None, help=None, on_change=None, args=None, kwargs=None,disabled=False, horizontal=False, label_visibility="visible")
        if Is_SSD_ == "Yes":
            Storage_SSD_ = col2.selectbox("Storage SSD ",Storage_SSD, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
            SSD_Type_ = col3.selectbox("Storage SSD Type ",SSD_Type, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        else:
            Storage_HDD_ = st.selectbox("Storage HDD ",Storage_HDD, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

        Add_HDD = st.checkbox("Add HDD Storage", value=False, key="one", help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        if Add_HDD and Is_SSD_ == "Yes":
                Storage_HDD_ = st.selectbox("Storage HDD ",Storage_HDD, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")



        st.markdown("***")
        st.subheader("Choose product Memory specifications.")
        col1,col2,col3 = st.columns(3)
        Memory_capacity_ = col1.number_input("Memory_capacity", min_value=1, max_value=Memory_capacity_max, step=1)
        Memory_Type_ = col2.selectbox("Memory Type ",Memory_Type, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        Memory_Speed_ = col3.selectbox("Memory Speed",[2400.0,2666.0,2933.0,3000.0,3200.0,3600.0,4000.0,4400.0], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        st.markdown("***")
        st.subheader("Choose product Operating System specifications.")
        col1,col2 = st.columns(2)

        OS_Corporation_ = col1.selectbox("OS Corporation",OS_Corporation, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        if OS_Corporation_ == "Google":
            OS_Version_ = col2.selectbox("OS Version",[item for item in OS_Version if  'Chrome' in item.split()], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        elif OS_Corporation_ == "No Operating System":
            OS_Version_ = col2.selectbox("OS Version",[item for item in OS_Version if 'No Operating System' in item.split()], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        elif OS_Corporation_ == "Microsoft":
            OS_Version_ = col2.selectbox("OS Version",[item for item in OS_Version if 'Windows' in item.split()], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        elif OS_Corporation_ == "Apple":
            OS_Version_ = col2.selectbox("OS Version",["MacOS"], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        st.markdown("***")
        st.subheader("Choose product Screen specifications.")
        Screen_Size_= st.selectbox("Screen_Size",Screen_Size, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        col1,col2,col3,col4,col5 = st.columns(5)
        if Screen_Size_ != "No Screen":
            X_res_ = col1.selectbox("X res",[item for item in X_res if 0 != item], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
            Y_res_ = col2.selectbox("Y res",[item for item in Y_res if 0 != item], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
            Res_Type_ = col3.selectbox("Res Type",[item for item in Res_Type if "No Screen" not in item], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
            Screen_Tec_ = col4.selectbox("Screen_Tec",Screen_Tec, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
            PPI_ = col5.selectbox("PPI",PPI, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        col1,col2,col3 = st.columns(3)
        Touchscreen_ = col1.radio("Do You Need it Touchscreen?",Touchscreen, index=0, key=None, help=None, on_change=None, args=None, kwargs=None,disabled=False, horizontal=True, label_visibility="visible")
        WideScreen_ = col2.radio("Do You Need it WideScreen?",WideScreen, index=0, key=None, help=None, on_change=None, args=None, kwargs=None,disabled=False, horizontal=True, label_visibility="visible")
        WebCam_ = col3.radio("Do You Need WebCam?",WebCam, index=0, key=None, help=None, on_change=None, args=None, kwargs=None,disabled=False, horizontal=True, label_visibility="visible")
        st.markdown("***")
        st.subheader("Choose product Another specifications.")
        col1,col2 = st.columns(2)
        AC_Power_ = col1.number_input("AC Power" ,min_value=AC_Power.min(), max_value=AC_Power.max(), step=20)
        Power_Supply_W_ = col2.number_input("Power_Supply_W" ,min_value=Power_Supply_W_min, max_value=Power_Supply_W_max, step=100)
        col1,col2 = st.columns(2)
        Bluetooth_ = col1.radio("Do You Need Bluetooth?",Bluetooth, index=0, key=None, help=None, on_change=None, args=None, kwargs=None,disabled=False, horizontal=True, label_visibility="visible")
        Bluetooth_V_ = col2.selectbox("Bluetooth_V",Bluetooth_V, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        col1,col2,col3 = st.columns(3)
        Ethernet_ = col1.radio("Do You Need Ethernet?",Ethernet, index=0, key=None, help=None, on_change=None, args=None, kwargs=None,disabled=False, horizontal=True, label_visibility="visible")
        WiFi_Ver_ = col2.selectbox("WiFi Ver",WiFi_Ver, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        Thunderbolt_ = col3.radio("Do You Need Thunderbolt?",Thunderbolt, index=0, key=None, help=None, on_change=None, args=None, kwargs=None,disabled=False, horizontal=True, label_visibility="visible")
        col1,col2,col3 = st.columns(3)
        Type_C_Count_ = col1.number_input("Type_C_Count", min_value=Type_C_Count_min, max_value=Type_C_Count_max,step=1.0)
        Type_A_Count_ = col2.number_input("Type_A_Count", min_value=Type_A_Count_min, max_value=Type_A_Count_max,step=1.0)
        USB_C_Ver_ = col3.selectbox("USB_C_Ver",USB_C_Ver, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        col1,col2 = st.columns(2)
        Product_Type_ = 0
        Battery_Cell = 0
        def Replace_Yes_No(string):
            if string == "Yes":
                return 1
            else:
                return 0
        Data = {}
        a = Is_SSD_
        b = Touchscreen_
        c = WideScreen_
        d = Bluetooth_ 
        e = Ethernet_ 
        f = Thunderbolt_ 
        Is_SSD_ =""
        Touchscreen_ =""
        WideScreen_ = ""
        Bluetooth_ = ""
        Ethernet_  = ""
        Thunderbolt_ =""
        Is_SSD_ = Replace_Yes_No(Is_SSD_)
        Touchscreen_ = Replace_Yes_No(Touchscreen_) 
        WideScreen_ = Replace_Yes_No(WideScreen_)
        Bluetooth_ =Replace_Yes_No(Bluetooth_)
        Ethernet_ =Replace_Yes_No(Ethernet_)
        Thunderbolt_ =Replace_Yes_No(Thunderbolt_)

elif selection_type == "Labtop":
    for i in Row_Desktop_Data:
        user_value = i
        locals()[user_value] = 0

    Product_Type_ = 0
    st.title("How Much Labtop Computers Cost?")
    Brand = [key for key in Labtop_Reverse_Brand.keys()]
    Color = [key for key in Labtop_Reverse_Color.keys()]
    Date = df_Labtop.Date_First_Year.unique().astype("int")
    Style = [key for key in Labtop_Reverse_Product_Style.keys()]
    Usage = [key for key in Labtop_Reverse_Product_Usage.keys()]
    Weight_min = df_Labtop.Weight.min()
    Weight_max = df_Labtop.Weight.max()
    CPU_Brand = [key for key in Labtop_Reverse_CPU_Brand.keys()]
    CPU_Model = [key for key in Labtop_Reverse_CPU_Model.keys()] 
    CPU_Series = [key for key in Labtop_Reverse_CPU_Series.keys()] 
    CPU_Gen = [key for key in Labtop_Reverse_CPU_Gen.keys()] 
    CPU_Speed = df_Labtop.CPU_Speed.unique()
    CPU_Cores = df_Labtop.Cores.astype("str").apply(lambda x: x if x != "nan" else 6.0).unique()
    Core_Name = [key for key in Desktop_Reverse_Core_Name.keys()]
    GPU_Brand = [key for key in Labtop_Reverse_GPU_Brand.keys()]
    GPU_Model = [key for key in Labtop_Reverse_GPU_Model.keys()]
    Video_Memory = df_Labtop.Video_Memory.unique().astype("int")
    Is_SSD = ["Yes","No"]
    Storage_HDD = df_Labtop.Storage_HDD.unique().astype("int")
    Storage_SSD = df_Labtop.Storage_SSD.apply(lambda x: x * 1024 if x < 20 else x).unique()
    SSD_Type = [key for key in Labtop_Reverse_SSD_Type.keys()]
    Memory_capacity_min = df_Labtop.Memory_capacity.min().astype("int")
    Memory_capacity_max = df_Labtop.Memory_capacity.max().astype("int")
    Memory_Type = [key for key in Labtop_Reverse_Memory_Type.keys()]
    Memory_Speed = df_Labtop.Memory_Speed.unique()

    OS_Corporation = [key for key in Labtop_Reverse_OS_Corporation.keys()]
    OS_Version = [key for key in Labtop_Reverse_OS_Version.keys()]
    Screen_Size = df_Labtop.Screen_Size.apply(lambda x: x if pd.to_numeric(x) >= 23 else 0).unique()
    X_res = df_Labtop.X_res.unique().astype("int")
    Y_res = df_Labtop.Y_res.unique().astype("int")
    Res_Type = [key for key in Labtop_Reverse_Res_Type.keys()]
    Touchscreen = df_Labtop.Touchscreen.apply(lambda x: "Yes" if x == 1.0 else "No").unique().tolist()
    WideScreen = df_Labtop.Touchscreen.apply(lambda x: "Yes" if x == 1.0 else "No").unique().tolist()
    Screen_Tec = [key for key in Labtop_Reverse_Screen_Tec.keys()]
    WebCam = [key for key in Labtop_Reverse_WebCam.keys()]
    AC_Power = df_Labtop.AC_Power.unique().astype("int")
    Battery_Cell = df_Labtop.Battery_Cell.unique()
    Power_Supply_W_min = df_Labtop.AC_Power.min().astype("int")
    Power_Supply_W_max = df_Labtop.AC_Power.max().astype("int")
    Bluetooth = df_Labtop.Bluetooth.apply(lambda x: "Yes" if x == 1.0 else "No").unique()
    def Bluetooth_V(float):
        if str(float).startswith("5"):
            return 5
        elif str(float).startswith("3"):
            return 3
        elif str(float).startswith("2"):
            return 2
        else:
            return 4
    Bluetooth_V = df_Labtop.Bluetooth_V.apply(Bluetooth_V).unique()
    Ethernet = df_Labtop.Ethernet.apply(lambda x: "Yes" if x == 1.0 else "No").unique()
    WiFi_Ver = [key for key in Labtop_Reverse_WiFi_Ver.keys()]
    Thunderbolt = df_Labtop.Has_Thunderbolt.apply(lambda x: "Yes" if x == 1.0 else "No").unique()
    Type_C_Count_min = df_Labtop.Type_C_Count.unique().min()
    Type_C_Count_max = df_Labtop.Type_C_Count.unique().max()
    Type_A_Count_min = df_Labtop.Type_A_Count.unique().min()
    Type_A_Count_max = df_Labtop.Type_A_Count.unique().max()
    USB_C_Ver = [key for key in Labtop_Reverse_USB_C_Ver.keys()]
    PPI = df_Labtop.PPI.unique()

    #############################################################################################
    col1,col2 = st.columns(2)
    st.markdown("***")
    st.subheader("Choose product Main specifications.")
    Brand_ = col1.selectbox("What Brand do you prefer? ", Brand, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    Color_ = col2.selectbox("What Color do you prefer? ", Color, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    col1,col2 = st.columns(2)
    Date_ = col1.selectbox("Choose the date of manufacture", Date, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    Style_ = col2.selectbox("Computer Style ",[item for item in Style if "Tower" not in item] , index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    Usage_ = st.selectbox("Product_Usage",Usage, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    Weight_ = st.number_input("Weight", min_value=Weight_min, max_value=Weight_max, step=0.25)
    st.markdown("***")
    st.subheader("Choose product CPU specifications.")
    col1,col2,col3,col4 = st.columns(4)
    CPU_Brand_ = col1.selectbox("CPU Brand", CPU_Brand, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    if CPU_Brand_ == "Intel" or CPU_Brand_ == "Xeon":
            CPU_Model_ = col2.selectbox("CPU Model",[item for i,item in enumerate(["Core" , "Celeron" , "Xeon","Dual-Core"]) if  ["Core" , "Celeron" , "Xeon","Dual-Core"][i] in item], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

            CPU_Series_ = col3.selectbox("CPU Series ",[item for item in CPU_Series if "i" in item], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

    elif CPU_Brand_ == "AMD":
            CPU_Model_ = col2.selectbox("CPU Model",[item for i,item in enumerate(["Ryzen","Athlon"]) if  ["Ryzen","Athlon"][i] in item], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

            CPU_Series_ = col3.selectbox("CPU Series ",[item for item in CPU_Series if "Ryzen" in item], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    elif CPU_Brand_ == "Apple":
            CPU_Model_ = col2.selectbox("CPU Model",[item for i,item in enumerate(["M1"]) if  ["M1"][i] in item], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
            CPU_Series_ = "i5"


    CPU_Gen_ = col4.selectbox("CPU Gen ",CPU_Gen, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    col1,col2 = st.columns(2)
    CPU_Speed_ = col1.selectbox("CPU Speed ",CPU_Speed, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    CPU_Cores_ = col2.selectbox("CPU Cores",CPU_Cores, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    #####
    Core_Name_ = st.selectbox("CPU Core Name",Core_Name, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    st.markdown("***")
    st.subheader("Choose product GPU specifications.")
    col1,col2 = st.columns(2)

    GPU_Brand_ = col1.selectbox("GPU Brand", GPU_Brand, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")


    if GPU_Brand_ == "Intel":
            GPU_Model_ = col2.selectbox("GPU Model",[item for i,item in enumerate(["Graphics" , "HD","UHD"]) if  ["Graphics" , "HD","UHD"][i] in item], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

    elif GPU_Brand_ == "NVIDIA":
            GPU_Model_ = col2.selectbox("GPU Model",[item for i,item in enumerate( ["GeForce","Radeon","Quadro","Iris","T1200"]) if   ["GeForce","Radeon","Quadro","Iris","T1200"][i] in item], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

    elif GPU_Brand_ == "Apple":
            GPU_Model_ = col2.selectbox("GPU Model",[item for i,item in enumerate(["M1"]) if  ["M1"][i] in item], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    elif GPU_Brand_ == "Imagination":
            GPU_Model_ = col2.selectbox("GPU Model",[item for i,item in enumerate(["PowerVR"]) if  ["PowerVR"][i] in item], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    elif GPU_Brand_ == "ARM":
            GPU_Model_ = col2.selectbox("GPU Model",[item for i,item in enumerate(["Mali"]) if  ["Mali"][i] in item], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    Video_Memory_ =round(float(st.number_input("Video_Memory", min_value=1, max_value=int(Video_Memory.max()))))


    st.markdown("***")
    st.subheader("Choose product Storage specifications.")
    col1,col2,col3 = st.columns(3)
    Is_SSD_ = col1.radio("Do You Need SSD Storage?",Is_SSD, index=0, key=None, help=None, on_change=None, args=None, kwargs=None,disabled=False, horizontal=False, label_visibility="visible")
    if Is_SSD_ == "Yes":
        Storage_SSD_ = col2.selectbox("Storage SSD ",Storage_SSD, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        SSD_Type_ = col3.selectbox("Storage SSD Type ",SSD_Type, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

    else:
        Storage_HDD_ = st.selectbox("Storage HDD ",Storage_HDD, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")


    Add_HDD = st.checkbox("Add HDD Storage", value=False, key="one", help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    if Add_HDD and Is_SSD_ == "Yes":
                Storage_HDD_ = st.selectbox("Storage HDD ",Storage_HDD, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
            
    st.markdown("***")
    st.subheader("Choose product Memory specifications.")
    col1,col2,col3 = st.columns(3)
    Memory_capacity_ = col1.number_input("Memory_capacity", min_value=1, max_value=Memory_capacity_max, step=1)
    Memory_Type_ = col2.selectbox("Memory Type ",Memory_Type, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    Memory_Speed_ = col3.selectbox("Memory Speed",[2400.0,2666.0,2933.0,3000.0,3200.0,3600.0,4000.0,4400.0], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    st.markdown("***")
    col1,col2 = st.columns(2)

    OS_Corporation_ = col1.selectbox("OS Corporation",OS_Corporation, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    if OS_Corporation_ == "Google":
        OS_Version_ = col2.selectbox("OS Version",[item for item in OS_Version if  'Chrome' in item.split()], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    elif OS_Corporation_ == "No Operating System":
        OS_Version_ = col2.selectbox("OS Version",[item for item in OS_Version if 'No Operating System' in item.split()], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    elif OS_Corporation_ == "Microsoft":
        OS_Version_ = col2.selectbox("OS Version",[item for item in OS_Version if 'Windows' in item.split()], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    elif OS_Corporation_ == "Apple":
        OS_Version_ = col2.selectbox("OS Version",["MacOS"], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    st.markdown("***")
    Screen_Size_ = st.selectbox("Screen_Size",Screen_Size, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    st.subheader("Choose product Screen specifications.")
    col1,col2,col3,col4,col5 = st.columns(5)
    if Screen_Size_ != "No Screen":
        X_res_ = col1.selectbox("X res",[item for item in X_res if 0 != item], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        Y_res_ = col2.selectbox("Y res",[item for item in Y_res if 0 != item], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        PPI_ = col3.selectbox("PPI",[item for item in PPI if 0 != item], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        Res_Type_ = col4.selectbox("Res Type",[item for item in Res_Type if "No Screen" not in item], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
        Screen_Tec_ = col5.selectbox("Screen_Tec",Screen_Tec, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    col1,col2,col3 = st.columns(3)
    Touchscreen_ = col1.radio("Do You Need it Touchscreen?",Touchscreen, index=0, key=None, help=None, on_change=None, args=None, kwargs=None,disabled=False, horizontal=True, label_visibility="visible")
    WideScreen_ = col2.radio("Do You Need it WideScreen?",WideScreen, index=0, key=None, help=None, on_change=None, args=None, kwargs=None,disabled=False, horizontal=True, label_visibility="visible")
    WebCam_ = col3.radio("Do You Need WebCam?",WebCam, index=0, key=None, help=None, on_change=None, args=None, kwargs=None,disabled=False, horizontal=True, label_visibility="visible")
    st.markdown("***")
    st.subheader("Choose product Another specifications.")
    col1,col2,col3 = st.columns(3)
    AC_Power_ = col1.number_input("AC Power" ,min_value=AC_Power.min(), max_value=AC_Power.max(), step=20)
    Battery_Cell_ = col2.selectbox("Battery Cell",Battery_Cell, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    Power_Supply_W_ = col3.number_input("Power_Supply_W" ,min_value=Power_Supply_W_min, max_value=Power_Supply_W_max, step=100)
    col1,col2 = st.columns(2)
    Bluetooth_ = col1.radio("Do You Need Bluetooth?",Bluetooth, index=0, key=None, help=None, on_change=None, args=None, kwargs=None,disabled=False, horizontal=True, label_visibility="visible")
    Bluetooth_V_ = col2.selectbox("Bluetooth_V",Bluetooth_V, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    col1,col2,col3 = st.columns(3)
    Ethernet_ = col1.radio("Do You Need Ethernet?",Ethernet, index=0, key=None, help=None, on_change=None, args=None, kwargs=None,disabled=False, horizontal=True, label_visibility="visible")
    WiFi_Ver_ = col2.selectbox("WiFi Ver",WiFi_Ver, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    Thunderbolt_ = col3.radio("Do You Need Thunderbolt?",Thunderbolt, index=0, key=None, help=None, on_change=None, args=None, kwargs=None,disabled=False, horizontal=True, label_visibility="visible")
    col1,col2,col3 = st.columns(3)
    Type_C_Count_ = col1.number_input("Type_C_Count", min_value=Type_C_Count_min, max_value=Type_C_Count_max,step=1.0)
    Type_A_Count_ = col2.number_input("Type_A_Count", min_value=Type_A_Count_min, max_value=Type_A_Count_max,step=1.0)
    USB_C_Ver_ = col3.selectbox("USB_C_Ver",USB_C_Ver, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    st.markdown("***")


##############################################
def Replace_Yes_No(string):
    if string == "Yes":
        return 1
    else:
        return 0
Data = {}
a = Is_SSD_
b = Touchscreen_
c = WideScreen_
d = Bluetooth_ 
e = Ethernet_ 
f = Thunderbolt_ 
Is_SSD_ =""
Touchscreen_ =""
WideScreen_ = ""
Bluetooth_ = ""
Ethernet_  = ""
Thunderbolt_ =""
Is_SSD_ = Replace_Yes_No(Is_SSD_)
Touchscreen_ = Replace_Yes_No(Touchscreen_) 
WideScreen_ = Replace_Yes_No(WideScreen_)
Bluetooth_ =Replace_Yes_No(Bluetooth_)
Ethernet_ =Replace_Yes_No(Ethernet_)
Thunderbolt_ =Replace_Yes_No(Thunderbolt_)


for user_value in Row_Desktop_Data:
    Data[user_value] = locals()[user_value]
df = pd.DataFrame([Data])
st.subheader("Display The Entered Data")

gb = GridOptionsBuilder.from_dataframe(df)
gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
gb.configure_side_bar() #Add a sidebar
gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
gridOptions = gb.build()

grid_response = AgGrid(df,
    gridOptions=gridOptions,
    data_return_mode='AS_INPUT', 
    update_mode='MODEL_CHANGED', 
    fit_columns_on_grid_load=False,
    enable_enterprise_modules=True,
    height=350, 
    width='100%',
    reload_data=True
)









Row_Data1 = []
Row_Data = []
######### Transform Data For Desktop
if selection_type == "Desktop":
    Row_Data = []
    path = "../Machine Learning/Reverse_Label\Desktop"
    CSV_Files = sorted([f for f in listdir(path) if isfile(join(path, f))])
    Var_Names_ = []
    for i in CSV_Files:
        var = i.replace("(", "").replace(")", "").replace(".csv","").replace(" ", "_")
        locals()[var] = pd.read_csv(path+"/"+i)
        locals()[var] = locals()[var].to_numpy().tolist()
        locals()[var] = { k[0]: k[1] for k in locals()[var] }
        for user_value in Row_Desktop_Data:
            if user_value.strip("_") in var:
                for key,value in locals()[var].items():
                    if locals()[user_value] == key:
                        locals()[user_value] = value
                        Row_Data.append(locals()[user_value])
    Row_Mode = []                  
                    
    Row_Model = []
    for user_value in Data:
        if user_value == "CPU_Cores_":
                Row_Model.append(locals()[user_value])
        if locals()[user_value] == None:
            Row_Data.append(0)
        else:
            if isinstance(locals()[user_value], str):
                Row_Data.append(float(locals()[user_value]))
            else:
                #Row_Data.append(locals()[user_value])
                #st.markdown(f"{user_value} - {locals()[user_value]}")
                Row_Model.append(locals()[user_value])
                Row_Mode.append(user_value)




    #res = " \n ".join("{} {}".format(Row_Mode, Row_Model) for Row_Mode, Row_Model in zip(Row_Mode, Row_Model))
    #st.markdown(res)
  
    Model_Desktop = joblib.load("../Deployment/Model/Model_Desktop_RandomForest_Test_Acc_0.9811082513729688.h5")
    Model_Desktop_Scaller = joblib.load("../Deployment/Model/Scaler_Model_Desktop_RandomForest.h5")
    predict = round(Model_Desktop.predict(Model_Desktop_Scaller.transform([Row_Model]))[0],2)
    submit = st.button("Predict")
    if submit:
        result_from  = round(predict-500)
        result_to = round(predict+500)
        st.success(f"Predicted price for this Computer could be between {round(predict)} SR  To  {round(predict+500)} SR")
        st.markdown("***")
        st.success(f"Predicted price for this Computer could be between {round((predict)/3.76)} USD  To  {round((predict+500)/3.76)} USA")

else:
######### Transform Data For Labtop
    Row_Data = []
    path = "../Machine Learning/Reverse_Label/Labtop"
    CSV_Files = sorted([f for f in listdir(path) if isfile(join(path, f))])
    Var_Names_ = []
    for i in CSV_Files:
        var = i.replace("(", "").replace(")", "").replace(".csv","").replace(" ", "_")
        locals()[var] = pd.read_csv(path+"/"+i)
        locals()[var] = locals()[var].to_numpy().tolist()
        locals()[var] = { k[0]: k[1] for k in locals()[var] }
        for user_value in Row_Desktop_Data:
            if user_value.strip("_") in var:

                for key,value in locals()[var].items():
                    if locals()[user_value] == key:
                        locals()[user_value] = value
                        Row_Data.append(locals()[user_value])
                        
                    
    Row_Model = []
    for user_value in Data:
        if locals()[user_value] == None:
            Row_Data.append(0)
        else:
            if isinstance(locals()[user_value], str):
                Row_Data.append(float(locals()[user_value]))
            else:
                #Row_Data.append(locals()[user_value])
                #st.markdown(f"{user_value} - {locals()[user_value]}")
                Row_Model.append(locals()[user_value])


    #st.markdown(Row_Model)
    #st.markdown(len(Row_Model))
  
    Model_Labtop = joblib.load("Model/Model_Labtop_RandomForest_Test_Acc_0.9241508604602842.h5")
    Model_Labtop_Scaller = joblib.load("Model/Scaler_Model_Labtop_RandomForest.h5")
    predict = round(Model_Labtop.predict(Model_Labtop_Scaller.transform([Row_Model]))[0],2)

    submit = st.button("Predict")
    if submit:
        result_from  = round(predict-500)
        result_to = round(predict+500)
        st.success(f"Predicted price for this Computer could be between {round(predict)} SR  To  {round(predict+500)} SR")
        st.markdown("***")
        st.success(f"Predicted price for this Computer could be between {round((predict)/3.76)} USD  To  {round((predict+500)/3.76)} USA")
    











#####################################################################################

import io
def df_info(df):
    df.columns = df.columns.str.replace(' ', '_')
    buffer = io.StringIO() 
    df.info(buf=buffer)
    s = buffer.getvalue() 

    df_info = s.split('\n')

    counts = []
    names = []
    nn_count = []
    dtype = []
    for i in range(5, len(df_info)-3):
        line = df_info[i].split()
        counts.append(line[0])
        names.append(line[1])
        nn_count.append(line[2])
        dtype.append(line[4])

    df_info_dataframe = pd.DataFrame(data = {'#':counts, 'Column':names, 'Non-Null Count':nn_count, 'Data Type':dtype})
    return df_info_dataframe.drop('#', axis = 1)

def df_isnull(df):
    res = pd.DataFrame(df.isnull().sum()).reset_index()
    res['Percentage'] = round(res[0] / df.shape[0] * 100, 2)
    res['Percentage'] = res['Percentage'].astype(str) + '%'
    return res.rename(columns = {'index':'Column', 0:'Number of null values'})

def number_of_outliers(df):
    
    df = df.select_dtypes(exclude = 'object')
    
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    
    ans = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
    df = pd.DataFrame(ans).reset_index().rename(columns = {'index':'column', 0:'count_of_outliers'})
    return df

def space(num_lines=1):
    for _ in range(num_lines):
        st.write("")

def sidebar_space(num_lines=1):
    for _ in range(num_lines):
        st.sidebar.write("")


def sidebar_multiselect_container(massage, arr, key):
    
    container = st.sidebar.container()
    select_all_button = st.sidebar.checkbox("Select all for " + key + " plots")
    if select_all_button:
        selected_num_cols = container.multiselect(massage, arr, default = list(arr))
    else:
        selected_num_cols = container.multiselect(massage, arr, default = arr[0])

    return selected_num_cols    



import streamlit as st
import pandas as pd
import plotly.express as px



df_Desktop = pd.read_csv(r"CSV/Desktop.csv")#[["Title_Product","Brand","Model","Series","Color","Date_First_Year","Product_Style","Product_Usage","Weight","OS_Version","Product_Price_US","Product_Price_SR","Shipping_Price","Total_Price","Product_URL","Product_Images"]]
df_Desktop["Total_Price_"] = df_Desktop["Total_Price"]
df_Desktop.drop("Total_Price",axis=1,inplace=True)
df_Labtop = pd.read_csv(r"CSV/Labtop.csv")#[["Title_Product","Brand","Model","Series","Color","Date_First_Year","Product_Style","Product_Usage","Weight","OS_Version","Product_Price_US","Product_Price_SR","Shipping_Price","Total_Price","Product_URL","Product_Images"]]
df_Labtop["Total_Price_"] = df_Labtop["Total_Price"]
df_Labtop.drop("Total_Price",axis=1,inplace=True)
space()
Use_ADA = st.checkbox("Do you want to do some analysis?",value=False)
if Use_ADA:
    if selection_type == "Desktop":
        df = df_Desktop
        st.success("Desktop Computers Analysis")
    else:
        df = df_Labtop
        st.success("Labtop Computers Analysis")

    
    st.subheader('Dataframe:')
    n, m = df.shape
    st.write(f'<p style="font-size:130%">Dataset contains {n} rows and {m} columns.</p>', unsafe_allow_html=True)   
    st.dataframe(df)


    all_vizuals = ['Info', 'NA Info', 'Descriptive Analysis', 'Target Analysis', 
                   'Distribution of Numerical Columns', 'Count Plots of Categorical Columns', 
                   'Box Plots', 'Outlier Analysis', 'Variance of Target with Categorical Columns']
    sidebar_space(3)         
    vizuals = st.sidebar.multiselect("Choose which visualizations you want to see 👇", all_vizuals)

    if 'Info' in vizuals:
        st.subheader('Info:')
        c1, c2, c3 = st.columns([1, 2, 1])
        c2.dataframe(df_info(df))

    if 'NA Info' in vizuals:
        st.subheader('NA Value Information:')
        if df.isnull().sum().sum() == 0:
            st.write('There is not any NA value in your dataset.')
        else:
            c1, c2, c3 = st.columns([0.5, 2, 0.5])
            c2.dataframe(df_isnull(df), width=1500)
            space(2)
            

    if 'Descriptive Analysis' in vizuals:
        st.subheader('Descriptive Analysis:')
        st.dataframe(df.describe())
        
    if 'Target Analysis' in vizuals:
        st.subheader("Select target column:")    
        target_column = st.selectbox("", df.columns, index = len(df.columns) - 1)
    
        st.subheader("Histogram of target column")
        fig = px.histogram(df, x = target_column)
        c1, c2, c3 = st.columns([0.5, 2, 0.5])
        c2.plotly_chart(fig)


    num_columns = df.select_dtypes(exclude = 'object').columns
    cat_columns = df.select_dtypes(include = 'object').columns

    if 'Distribution of Numerical Columns' in vizuals:

        if len(num_columns) == 0:
            st.write('There is no numerical columns in the data.')
        else:
            selected_num_cols = sidebar_multiselect_container('Choose columns for Distribution plots:', num_columns, 'Distribution')
            st.subheader('Distribution of numerical columns')
            i = 0
            while (i < len(selected_num_cols)):
                c1, c2 = st.columns(2)
                for j in [c1, c2]:

                    if (i >= len(selected_num_cols)):
                        break

                    fig = px.histogram(df, x = selected_num_cols[i])
                    j.plotly_chart(fig, use_container_width = True)
                    i += 1

    if 'Count Plots of Categorical Columns' in vizuals:

        if len(cat_columns) == 0:
            st.write('There is no categorical columns in the data.')
        else:
            selected_cat_cols = sidebar_multiselect_container('Choose columns for Count plots:', cat_columns, 'Count')
            st.subheader('Count plots of categorical columns')
            i = 0
            while (i < len(selected_cat_cols)):
                c1, c2 = st.columns(2)
                for j in [c1, c2]:

                    if (i >= len(selected_cat_cols)):
                        break

                    fig = px.histogram(df, x = selected_cat_cols[i], color_discrete_sequence=['indianred'])
                    j.plotly_chart(fig)
                    i += 1

    if 'Box Plots' in vizuals:
        if len(num_columns) == 0:
            st.write('There is no numerical columns in the data.')
        else:
            selected_num_cols = sidebar_multiselect_container('Choose columns for Box plots:', num_columns, 'Box')
            st.subheader('Box plots')
            i = 0
            while (i < len(selected_num_cols)):
                c1, c2 = st.columns(2)
                for j in [c1, c2]:
                    
                    if (i >= len(selected_num_cols)):
                        break
                    
                    fig = px.box(df, y = selected_num_cols[i])
                    j.plotly_chart(fig, use_container_width = True)
                    i += 1

    if 'Outlier Analysis' in vizuals:
        st.subheader('Outlier Analysis')
        c1, c2, c3 = st.columns([1, 2, 1])
        c2.dataframe(number_of_outliers(df))

    if 'Variance of Target with Categorical Columns' in vizuals:
        
        
        df_1 = df.dropna()
        
        high_cardi_columns = []
        normal_cardi_columns = []

        for i in cat_columns:
            if (df[i].nunique() > df.shape[0] / 10):
                high_cardi_columns.append(i)
            else:
                normal_cardi_columns.append(i)


        if len(normal_cardi_columns) == 0:
            st.write('There is no categorical columns with normal cardinality in the data.')
        else:
        
            st.subheader('Variance of target variable with categorical columns')
            model_type = st.radio('Select Problem Type:', (['Regression']), key = 'model_type')
            selected_cat_cols = sidebar_multiselect_container('Choose columns for Category Colored plots:', normal_cardi_columns, 'Category')
            
            if 'Target Analysis' not in vizuals:   
                target_column = st.selectbox("Select target column:", df.columns, index = len(df.columns) - 1)
            
            i = 0
            while (i < len(selected_cat_cols)):
                
                
            
                if model_type == 'Regression':
                    fig = px.box(df_1, y = target_column, color = selected_cat_cols[i])
                else:
                    fig = px.histogram(df_1, color = selected_cat_cols[i], x = target_column)

                st.plotly_chart(fig, use_container_width = True)
                i += 1

            if high_cardi_columns:
                if len(high_cardi_columns) == 1:
                    st.subheader('The following column has high cardinality, that is why its boxplot was not plotted:')
                else:
                    st.subheader('The following columns have high cardinality, that is why its boxplot was not plotted:')
                for i in high_cardi_columns:
                    st.write(i)
                
                st.write('<p style="font-size:140%">Do you want to plot anyway?</p>', unsafe_allow_html=True)    
                answer = st.selectbox("", ('No', 'Yes'))

                if answer == 'Yes':
                    for i in high_cardi_columns:
                        fig = px.box(df_1, y = target_column, color = i)
                        st.plotly_chart(fig, use_container_width = True)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
st.caption("By: Waleed Hassan - Eng.Waleedhassan11@gmail.com -Epsilon Data Scince Project - 18-12-2022")