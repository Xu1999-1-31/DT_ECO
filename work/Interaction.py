import subprocess
import shutil
import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
import Global_var

def check_command(command_name):
    # check if the command is available
    return shutil.which(command_name) is not None

def Run_Pt_Script(script):
    if not check_command('pt_shell'):
        raise EnvironmentError("Error: 'pt_shell' not found. Please ensure PrimeTime is correctly installed and added to your PATH.")
    command = ['pt_shell', '-f', '../' + script]
    working_directory = os.path.join(Global_var.work_dir, 'log/')
    if not os.path.exists(working_directory):
        os.makedirs(working_directory)
    print(f'Running PT script: {script} in {working_directory}')
    with subprocess.Popen(command, cwd=working_directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True) as process:
        for line in process.stdout:
            print(line, end='')  # cmd output     
        for err_line in process.stderr:
            print(err_line, end='')  # cmd err
        # stdout, stderr = process.communicate()
        
        # wait for the process to exit
        process.wait()

    # print(stdout)
    # print(stderr)
    
    # check the return code
    print('Return code:', process.returncode)

def Run_Icc2_Script(script):
    if not check_command('icc2_shell'):
        raise EnvironmentError("Error: 'icc2_shell' not found. Please ensure IC_Compiler2 is correctly installed and added to your PATH.")
    command = ['icc2_shell', '-f', '../' + script]
    working_directory = os.path.join(Global_var.work_dir, 'log/')
    if not os.path.exists(working_directory):
        os.makedirs(working_directory)
    print(f'Running ICC2 script: {script} in {working_directory}')
    with subprocess.Popen(command, cwd=working_directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True) as process:
        for line in process.stdout:
            print(line, end='')  # cmd output     
        for err_line in process.stderr:
            print(err_line, end='')  # cmd err
        pass
        
        # wait for the process to exit
        process.wait()

    # check the return code
    print('Return code:', process.returncode)

def Write_Pt_Scripts(design, ECO=True, VerilogInline=False, verbose=False):
    if ECO and VerilogInline:
        raise ValueError("ECO and VerilogInline cannot both be True at the same time.")
    # read initial file
    if VerilogInline:
        file = os.path.join(Global_var.work_dir, 'pt_rpt_inline.tcl')
    else:
        file = os.path.join(Global_var.work_dir, 'pt_rpt.tcl')
    with open(file, 'r') as file:
        script_content = file.read()

    # replace initial design to new design
    if ECO:
        updated_content = script_content.replace('set top_design aes_cipher_top', f'set top_design {design}_eco')
        updated_content = updated_content.replace('../Delay_scripts/${top_design}_Delay.tcl', f'../Delay_scripts/{design}_Delay.tcl')
    else:
        updated_content = script_content.replace('set top_design aes_cipher_top', f'set top_design {design}')

    # write new file
    script_filename = os.path.join(Global_var.work_dir, f'{design}_pt_rpt.tcl')
    with open(script_filename, 'w') as new_file:
        new_file.write(updated_content)
    
    if verbose:
        print(f"Generated PT script for {design}")
        
def Write_Icc2_Scripts(design, verbose=False):
    # read initial file
    file = os.path.join(Global_var.work_dir, 'icc2_rpt.tcl')
    with open(file, 'r') as file:
        script_content = file.read()

    # replace initial design to new design
    updated_content = script_content.replace('set bench aes_cipher_top', f'set bench {design}')

    # write new file
    script_filename = os.path.join(Global_var.work_dir, f'{design}_icc2_rpt.tcl')
    with open(script_filename, 'w') as new_file:
        new_file.write(updated_content)
    
    if verbose:
        print(f"Generated ICC2 script for {design}")

def Write_Icc2_ECO_Scripts(design, verbose=False):
    # read initial file
    file = os.path.join(Global_var.work_dir, 'icc2_eco.tcl')
    with open(file, 'r') as file:
        script_content = file.read()

    # replace initial design to new design
    updated_content = script_content.replace('set bench aes_cipher_top', f'set bench {design}')

    # write new file
    script_filename = os.path.join(Global_var.work_dir, f'{design}_icc2_eco.tcl')
    with open(script_filename, 'w') as new_file:
        new_file.write(updated_content)
    
    if verbose:
        print(f"Generated ICC2 ECO script for {design}")

def Delete_Temp_Scripts(design, verbose=False):
    path = os.path.join(Global_var.work_dir, f'{design}_icc2_rpt.tcl')
    if os.path.exists(path):
        os.remove(path)
    path = os.path.join(Global_var.work_dir, f'{design}_icc2_eco.tcl')
    if os.path.exists(path):
        os.remove(path)
    path = os.path.join(Global_var.work_dir, f'{design}_pt_rpt.tcl')
    if os.path.exists(path):
        os.remove(path)
    if verbose:
        print(f"Deleted temporary scripts for {design}")
        
def ECO_PRPT_Iteration(design, verbose=False): # one eco iteriation: incremental PR and PT
    Write_Icc2_ECO_Scripts(design, verbose)
    Run_Icc2_Script(f'{design}_icc2_eco.tcl')
    Write_Pt_Scripts(design, True, False, verbose) # Write PT script for ECO
    Run_Pt_Script(f'{design}_pt_rpt.tcl')
    Delete_Temp_Scripts(design, verbose)

def VerilogInline_PT_Iteration(design, verbose=False):
    Write_Pt_Scripts(design, False, True, verbose) # Write PT script for Verilog Inline change
    Run_Pt_Script(f'{design}_pt_rpt.tcl')
    Delete_Temp_Scripts(design, verbose)

def VerilogInlineChange(design, cells, Incremental=False, verbose=False): # cells:[name: [initial cell, final cell]]
    # read initial file
    if Incremental:
        file = os.path.join(Global_var.work_dir, 'VerilogInline/' + design + '_route.v')
    else:
        file = os.path.join(Global_var.work_dir, 'Icc2Output/' + design + '_route.v')
    with open(file, 'r') as infile:
        content = infile.read()
    for key, value in cells.items():
        content = content.replace(value[0]+' '+key+' (', value[1]+' '+key+' (')
    file = os.path.join(Global_var.work_dir, 'VerilogInline/' + design + '_route.v')
    with open(file, 'w') as outfile:
        outfile.write(content)
    if verbose:
        print(f"Changed Verilog Inline for {design}")

def  Write_Incremental_ECO_Scripts(design, cellLists, verbose=False): # write changelist for icc2; cellLists: [{cellName: [initial cell, final cell], ...}, ...]
    path = os.path.join(Global_var.work_dir, f'ECO_ChangeList/{design}_dt_eco.tcl')
    with open(path, 'w') as file:
        file.write(f'current_instance\n')
        for celldict in cellLists:
            for key, value in celldict.items():
                file.write(f'size_cell {{{key}}} {{{value[1]}}}\n')
    if verbose:
        print(f"Generated Incremental ECO script for {design}")