from typing import List, Dict
from json import loads
import numpy as np
from cvxopt import matrix
from cvxopt.solvers import qp

# Put free variables from details and subtotals arrays into a single array
def get_variables_mapping(table_details: List[List[int]], country_subtotals: List[int], sector_subtotals: List[int]):
  variables_mapping: List[Dict[str, int]] = []
  for i in range(len(table_details)):
    for j in range(len(table_details[0])):
      if table_details[i][j] == None:
        variables_mapping.append({"i":i,"j":j})
  
  for i in range(len(country_subtotals)):
    if country_subtotals[i] == None:
      variables_mapping.append({"i":i, "j": None})

  for j in range(len(sector_subtotals)):
    if sector_subtotals[j] == None:
      variables_mapping.append({"j":j, "i": None})

  return variables_mapping

def get_variable_index(i: int or None, j: int or None, variables_mapping: List[Dict[str, int]]):
  return variables_mapping.index({"i": i, "j": j})


# Parse altitude input into a format understable by cvxopt
def parse_input(table):
  table_details: List[List[int]] = table["details"]
  country_subtotals: List[int] = table["subtotals"]["byCountry"]
  sector_subtotals: List[int] = table["subtotals"]["bySector"]
  total: int = table["total"]


  variables_mapping = get_variables_mapping(table_details, country_subtotals, sector_subtotals)
  variables_count = len(variables_mapping)

  coeffs: List[List[float]] = []
  subtotals: List[float] = []

  for row in range(len(table_details)):
    if all(map(lambda v: v != None, table_details[row])) and country_subtotals[row] != None:
      continue
      
    coeffs.append([0.] * variables_count)
    subtotals.append(0.)
    for col in range(len(table_details[0])):
      if table_details[row][col] == None:
        coeffs[-1][get_variable_index(row, col, variables_mapping)] = 1.
      else:
        subtotals[-1] -= table_details[row][col]
    
    if country_subtotals[row] == None:
      coeffs[-1][get_variable_index(row, None, variables_mapping)] = -1.
    else:
      subtotals[-1] += country_subtotals[row]

  for col in range(len(table_details[0])):
    if all(map(lambda v: v != None, list(map(lambda n,c=col: n[c], table_details)))) and sector_subtotals[col] != None:
      continue
      
    coeffs.append([0.] * variables_count)
    subtotals.append(0.)
    for row in range(len(table_details)):
      if table_details[row][col] == None:
        coeffs[-1][get_variable_index(row, col, variables_mapping)] = 1.
      else:
        subtotals[-1] -= table_details[row][col]
    
    if sector_subtotals[col] == None:
      coeffs[-1][get_variable_index(None, col, variables_mapping)] = -1.
    else:
      subtotals[-1] += sector_subtotals[col]

  if any(map(lambda v: v == None, country_subtotals)):
    coeffs.append([0.] * variables_count)
    subtotals.append(float(total))
    for row in range(len(country_subtotals)):
      if country_subtotals[row] == None:
        coeffs[-1][get_variable_index(row, None, variables_mapping)] = 1.
      else:
        subtotals[-1] -= country_subtotals[row]

  if any(map(lambda v: v == None, sector_subtotals)):
    coeffs.append([0.] * variables_count)
    subtotals.append(float(total))
    for col in range(len(sector_subtotals)):
      if sector_subtotals[col] == None:
        coeffs[-1][get_variable_index(None, col, variables_mapping)] = 1.
      else:
        subtotals[-1] -= sector_subtotals[col]

  return coeffs, subtotals


# Removes excess matrix rows s.t. nb of rows = matrix rank
def get_reduced_matrix(m: List[List[float]], t: List[float]):
  m_copy = [m[i].copy() for i in range(len(m))]
  t_copy = t.copy()
  rank = np.linalg.matrix_rank(m_copy)
  k = len(m_copy) - 1

  while rank != len(m_copy):
    new_m = [m_copy[i].copy() for i in range(len(m_copy)) if i != k]
    new_t = [t_copy[i] for i in range(len(t_copy)) if i != k]
    new_rank = np.linalg.matrix_rank(new_m)

    if new_rank == rank:
      m_copy = new_m
      t_copy = new_t
    k -= 1

  return m_copy, t_copy


def get_solution(coeffs: List[List[int]], subtotals: List[int]) -> List[float]:

  n = len(coeffs[0])
  _R = matrix(np.zeros(n))
  _Q = matrix(np.eye(n))
  _G = matrix(- np.eye(n))
  _H = matrix(np.zeros(n))

  A = matrix(coeffs).T
  B = matrix(subtotals)

  return qp(_Q, -_R, _G, _H, A, B)['x']


def validate_input(user_input):
  pass


def adjust_output(solution: List[float], user_input):
  integer_solution = list(map(lambda f: int(f), solution))
  variables_mapping = get_variables_mapping(
    user_input["details"],
    user_input["subtotals"]["byCountry"],
    user_input["subtotals"]["bySector"])

  for index,v in enumerate(variables_mapping):
    i = v["i"]
    j = v["j"]

    if i == None:
      user_input["subtotals"]["bySector"][j] = integer_solution[index]
      continue

    if j == None:
      user_input["subtotals"]["byCountry"][i] = integer_solution[index]
      continue
    
    user_input["details"][i][j] = integer_solution[index]

  print("Before adjustment", user_input)

  if rec(variables_mapping, len(variables_mapping) - 1, user_input) == True:
    return user_input
  
  raise 'Failed to adjust'


def rec(variables, variable_index, user_input):
  if variable_index == -1:
    return True

  variable = variables[variable_index]

  if variable["i"] == None:
    is_free_on_y = any(map(lambda v: v["j"] == variable["j"] and v["i"] != None, variables))
    is_free_on_x = any(map(lambda v: v["j"] != variable["j"] and v["i"] == None, variables))

    y_possibilities = [0, 1] if is_free_on_y else [user_input["subtotals"]["bySector"][variable["j"]] - sum(map(lambda v: v[variable["j"]], user_input["details"]))]
    x_possibilities = [0, 1] if is_free_on_x else [user_input["total"] - sum(user_input["subtotals"]["bySector"])]

    if y_possibilities[0] > 1 or x_possibilities[0] > 1:
      return False

    possibilities = [
      *([0] if 0 in x_possibilities and 0 in y_possibilities else []), 
      *([1] if 1 in x_possibilities and 1 in y_possibilities else [])
    ]

    if len(possibilities) == 0:
      return False
    
    for p in possibilities:
      user_input["subtotals"]["bySector"][variable["j"]] += p

      if rec(variables, variable_index - 1, user_input):
        print("OK sector", p)
        return True

      print("KO sector", p)
      user_input["subtotals"]["bySector"][variable["j"]] -= p
    
    return False


  if variable["j"] == None:
    is_free_on_y = any(map(lambda v: v["i"] != variable["i"] and v["j"] == None, variables))
    is_free_on_x = any(map(lambda v: v["i"] == variable["i"] and v["j"] != None, variables))

    x_possibilities = [0, 1] if is_free_on_x else [user_input["subtotals"]["byCountry"][variable["i"]] - sum(user_input["details"][variable["i"]])]
    y_possibilities = [0, 1] if is_free_on_y else [user_input["total"] - sum(user_input["subtotals"]["byCountry"])]

    if y_possibilities[0] > 1 or x_possibilities[0] > 1:
      return False

    possibilities = [
      *([0] if 0 in x_possibilities and 0 in y_possibilities else []), 
      *([1] if 1 in x_possibilities and 1 in y_possibilities else [])
    ]

    if len(possibilities) == 0:
      return False
    
    for p in possibilities:
      user_input["subtotals"]["byCountry"][variable["i"]] += p

      if rec(variables, variable_index - 1, user_input):
        print("OK country", p)
        return True

      print("KO country", p)
      user_input["subtotals"]["byCountry"][variable["i"]] -= p
    
    return False


  else:
    is_free_on_y = any(map(lambda v: v["i"] != variable["i"] and v["j"] == variable["j"], variables))
    is_free_on_x = any(map(lambda v: v["i"] == variable["i"] and v["j"] != variable["j"], variables))

    x_possibilities = [0, 1] if is_free_on_x else [user_input["subtotals"]["byCountry"][variable["i"]] - sum(user_input["details"][variable["i"]])]
    y_possibilities = [0, 1] if is_free_on_y else [user_input["subtotals"]["bySector"][variable["j"]] - sum(map(lambda v: v[variable["j"]], user_input["details"]))]

    if y_possibilities[0] > 1 or x_possibilities[0] > 1:
      return False

    possibilities = [
      *([0] if 0 in x_possibilities and 0 in y_possibilities else []), 
      *([1] if 1 in x_possibilities and 1 in y_possibilities else [])
    ]

    if len(possibilities) == 0:
      return False
    
    for p in possibilities:
      user_input["details"][variable["i"]][variable["j"]] += p

      if rec(variables, variable_index - 1, user_input):
        print("OK details", p)
        
        return True

      print("KO details", p)
      user_input["details"][variable["i"]][variable["j"]] -= p
    
    return False


def lambda_handler(event, _):
  user_input = loads(event["body"])
  validate_input(user_input)

  parsed_input = parse_input(user_input)
  reduced_input = get_reduced_matrix(*parsed_input)
  solution = get_solution(*reduced_input)
  
  adjust_output(solution, user_input)
  
  return user_input

user_input = {
  "body": '''{
    "total": 70,
    "details": [
        [null, null, 20, null, null],
        [null, null, null, null, null],
        [10, null, null, null, null],
        [null, null, null, null, null],
        [null, null, null, null, null]
    ],
    "subtotals": {
      "byCountry": [null, null, null, 15, null],
      "bySector": [null, null, null, null, null]
    }
  }'''
}

print(lambda_handler(user_input, 'context'))