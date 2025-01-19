from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import google.generativeai as genai
from dotenv import load_dotenv
import os
from typing import Annotated
from fastapi import Depends, FastAPI, HTTPException, Query
from sqlmodel import Field, Session, SQLModel, create_engine, select

load_dotenv()  # este metodo carga las variables del fichero .env.

class Googlechat(BaseModel):
    """ prompt: str """ #usado para el chat con gemini api
    ingredients: Optional[list] = None

class Recipe(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    description: str

db_password = os.environ["DB_PASSWORD"]
db_host = os.environ["DB_HOST"]
db_port = os.environ["DB_PORT"]
db_username = os.environ["DB_USERNAME"]
db_name = os.environ["DB_NAME"]

db_url = f"postgresql+psycopg2://{db_username}:{db_password}@{db_host}/{db_name}"




""" sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}" """

#connect_args = {"check_same_thread": False}
connect_args = { }
engine = create_engine(db_url, connect_args=connect_args)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]

app = FastAPI()

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

@app.get("/recipes/")
async def recipe_list(
    session: SessionDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 100,
) -> list[Recipe]:
    recipes = session.exec(select(Recipe).offset(offset).limit(limit)).all()
    return recipes


#ejemplo de creador de recetas con gemini api
@app.post("/recipe/")
async def recipe_maker(prompt: Googlechat, session: SessionDep)-> Recipe:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")
    my_recipe_prompt = """
eres un cocinero de varias estrellas michelin.

te voy a dar un listado de ingredientes y lo que debes generar es una receta de alguna estrella michelin que contenga estos ingredientes.
los ingredientes son los siguientes:

%s

aqui te paso un ejemplo de receta:

pasos de la receta:
tiempo: 3 minutos
paso 1: ingredientes listos
paso 2: pon un hilito de aove en una sarten y echa los huevos un poco batidos. Echa un poco de atun. Remueve despacio.
paso 3: en cuanto cuaje el huevo, pon una ramita de oregano y sirvelo.
ingredientes: huevo y atun.

devuelmeme la receta en formato markdown.


""" %(prompt.ingredients)
    response = model.generate_content(my_recipe_prompt)

    json_compatible_item_data = jsonable_encoder(response.text)

    recipe = Recipe()
    recipe.description = response.text
    session.add(recipe)
    session.commit()
    session.refresh(recipe)

    return JSONResponse(content=json_compatible_item_data)
    return {"response": response.json()}
    #return {"response": prompt}

